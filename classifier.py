#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier on SST and CFIMDB
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from evaluation import model_eval_sentiment, model_test_sentiment
from tqdm import tqdm

from utils import sync_if_cuda
import time
from datetime import datetime
import json
from pathlib import Path
import copy

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class GPT2SentimentClassifier(torch.nn.Module):
  '''
  This module performs sentiment classification using GPT2 in a cloze-style (fill-in-the-blank) task.

  In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
  Thus, your forward() should return one logit for each of the 5 classes.
  '''

  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()

    # Pretrain mode does not require updating GPT paramters.
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    ### TODO: Create any instance variables you need to classify the sentiment of BERT embeddings.
    ### YOUR CODE HERE
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.projection = torch.nn.Linear(config.hidden_size, config.num_labels)


  def forward(self, input_ids, attention_mask):
    '''Takes a batch of sentences and returns logits for sentiment classes'''

    ### TODO: The final GPT contextualized embedding is the hidden state of the last token.
    ###       HINT: You should consider what is an appropriate return value given that
    ###       the training loop currently uses F.cross_entropy as the loss function.
    ### YOUR CODE HERE

    last_hidden = self.gpt(input_ids, attention_mask)["last_token"]
    last_droped = self.dropout(last_hidden)
    logits = self.projection(last_droped)

    return logits



class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


class SentimentTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    sent_ids = [x[1] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
  num_labels = {}
  data = []
  if flag == 'test':
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data


def save_model(model, optimizer, args, config, filepath):

  Path(filepath).parent.mkdir(parents=True, exist_ok=True)
  
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

  # Init model.
  config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            'num_labels': num_labels,
            'hidden_size': args.d,
            'data_dir': '.',
            'fine_tune_mode': args.fine_tune_mode}

  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  best_dev_acc = 0
  args.best_epoch = 0
  epochs_without_improvement = 0

  # Run for the specified number of epochs.
  sync_if_cuda()
  start = time.time()

  if args.use_gpu:
    torch.cuda.reset_peak_memory_stats()

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / (num_batches)

    train_acc, train_f1, *_ = model_eval_sentiment(train_dataloader, model, device)
    dev_acc, dev_f1, *_ = model_eval_sentiment(dev_dataloader, model, device)


    ## Early stopping 
    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      args.best_epoch = epoch
      save_model(model, optimizer, args, config, args.filepath)
      epochs_without_improvement = 0 

    else:
      epochs_without_improvement += 1

    if epochs_without_improvement >= args.patience:
      print(f"Early stopping at epoch {epoch}")
      print(f"Best epoch was {args.best_epoch}")
      break

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

  # Save training time
  sync_if_cuda()
  args.train_time = time.time() - start

  # Save memory usage
  if args.use_gpu:
    args.train_peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    args.train_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9


def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, weights_only=False)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    sync_if_cuda()
    start = time.time()

    if args.use_gpu:
      torch.cuda.reset_peak_memory_stats()

    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval_sentiment(dev_dataloader, model, device)
    print('DONE DEV')

    test_pred, test_sents, test_sent_ids = model_test_sentiment(test_dataloader, model, device)
    print('DONE Test')

    sync_if_cuda()
    args.evaluation_time = time.time() - start

    if args.use_gpu:
      args.eval_peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
      args.eval_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9


    ## Saving
    for path in [args.dev_out, args.test_out, args.summary_path]:
      Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")

    with open(args.summary_path, "w") as f:
      data = {"dev_accuracy": dev_acc, **vars(args)}
      json.dump(data, f, indent=2)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--sst_batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
  parser.add_argument("--cfimdb_batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  add_arguments(args)

  ### SST

  sst_args = copy.copy(args)
  sst_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
  sst_args.filepath=f'checkpoints/{args.model_size}-sst-classifier.pt'
  sst_args.train='data/ids-sst-train.csv'
  sst_args.dev='data/ids-sst-dev.csv'
  sst_args.test='data/ids-sst-test-student.csv'
  sst_args.dev_out=f"sentiment_results/{sst_timestamp}/sst_dev_out.csv"
  sst_args.test_out=f"sentiment_results/{sst_timestamp}/sst_test_out.csv"
  sst_args.summary_path=f"sentiment_results/{sst_timestamp}/sst_summary.json"
  sst_args.batch_size = args.sst_batch_size

  print('Training Sentiment Classifier on SST...')
  train(sst_args)

  print('Evaluating on SST...')
  test(sst_args)

  ### CFIMDB
  cfimdb_args = copy.copy(args)
  cfimdb_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
  cfimdb_args.filepath=f'checkpoints/{args.model_size}-cfimdb-classifier.pt'
  cfimdb_args.train='data/ids-cfimdb-train.csv'
  cfimdb_args.dev='data/ids-cfimdb-dev.csv'
  cfimdb_args.test='data/ids-cfimdb-test-student.csv'
  cfimdb_args.dev_out=f"sentiment_results/{cfimdb_timestamp}/cfimdb_dev_out.csv"
  cfimdb_args.test_out=f"sentiment_results/{cfimdb_timestamp}/cfimdb_test_out.csv"
  cfimdb_args.summary_path=f"sentiment_results/{cfimdb_timestamp}/cfimdb_summary.json"
  cfimdb_args.batch_size = args.cfimdb_batch_size

  print('Training Sentiment Classifier on cfimdb...')
  train(cfimdb_args)

  print('Evaluating on cfimdb...')
  test(cfimdb_args)
