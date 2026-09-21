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
import torch.distributed as dist
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from evaluation import model_eval_sentiment, model_test_sentiment, plot_training
from tqdm import tqdm
from datasets import load_sentiment_data, SentimentDataset, SentimentTestDataset

from utils import sync_if_cuda, flush_memory, seed_everything, save_model, add_size_arguments
import time
from datetime import datetime
import json
from pathlib import Path
import copy

TQDM_DISABLE = False


class GPT2SentimentClassifier(torch.nn.Module):
  '''
  This module performs sentiment classification using GPT2 in a cloze-style (fill-in-the-blank) task.

  In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
  Thus, your forward() should return one logit for each of the 5 classes.
  '''

  def __init__(self, args):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = args.num_labels
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)

    # Pretrain mode does not require updating GPT paramters.
    assert args.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if args.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif args.fine_tune_mode == 'full-model':
        param.requires_grad = True

    ### TODO: Create any instance variables you need to classify the sentiment of BERT embeddings.
    ### YOUR CODE HERE
    self.dropout = torch.nn.Dropout(args.hidden_dropout_prob)
    self.projection = torch.nn.Linear(args.d, args.num_labels)


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


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if args.use_gpu:
    torch.cuda.reset_peak_memory_stats()

  # Create the data and its corresponding datasets and dataloader.
  train_data, num_labels = load_sentiment_data(args.train, 'train')
  dev_data = load_sentiment_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

  # Init model.
  args.num_labels = num_labels

  model = GPT2SentimentClassifier(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
  optimizer.zero_grad()
  best_dev_acc = 0
  args.best_epoch = 0
  epochs_without_improvement = 0
  scaler = torch.amp.GradScaler('cuda', enabled=args.use_gpu)
  train_loss_history = []
  dev_acc_history = []

  # Run for the specified number of epochs.
  sync_if_cuda()
  start = time.time()

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


      # Mixed Precision training on GPU
      if args.use_gpu:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
          logits = model(b_ids, b_mask)
          loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size / args.grad_accum_steps
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (num_batches + 1) % args.grad_accum_steps == 0:
          scaler.step(optimizer)
          optimizer.zero_grad()
          scaler.update()

      else:
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size / args.grad_accum_steps
        loss.backward()

        # Gradient accumulation
        if (num_batches + 1) % args.grad_accum_steps == 0:
          optimizer.step()
          optimizer.zero_grad()

      train_loss += loss.item()
      num_batches += 1

    # Loss has been divideb by acumm_steps to normalize the gradient that will acumulate, so now need to rescale
    train_loss = train_loss * args.grad_accum_steps / num_batches
    train_loss_history.append(train_loss)

    train_acc, train_f1, *_ = model_eval_sentiment(train_dataloader, model, device)
    dev_acc, dev_f1, *_ = model_eval_sentiment(dev_dataloader, model, device)
    dev_acc_history.append(dev_acc)

    ## Early stopping 
    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      args.best_epoch = epoch
      save_model(model, optimizer, args, args.filepath)
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
  args.train_time = (time.time() - start) / 60

  # Save memory usage
  if args.use_gpu:
    args.train_peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    args.train_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

  plot_training(train_loss_history, dev_acc_history, "Accuracy", args.plot_path)
  

def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    if args.use_gpu:
      torch.cuda.reset_peak_memory_stats()

    saved = torch.load(args.filepath, weights_only=False)
    model = GPT2SentimentClassifier(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    dev_data = load_sentiment_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    test_data = load_sentiment_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    sync_if_cuda()
    start = time.time()

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

    with open(args.summary_path, "a") as f:
      data = {"dev_accuracy": dev_acc, **vars(args)}
      f.write(json.dumps(data) + "\n")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", default=None, choices=("sst", "cfimdb"), help="Dataset to train on.")
  
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--keep_model_checkpoint", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
  parser.add_argument("--grad_accum_steps", help='Accumulation steps for gradient updates.', type=int, default=1)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-5)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  add_size_arguments(args)
  
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
  args.filepath=f'checkpoints/{args.model_size}-{args.mode}-classifier.pt'
  args.train=f'data/ids-{args.mode}-train.csv'
  args.dev=f'data/ids-{args.mode}-dev.csv'
  args.test=f'data/ids-{args.mode}-test-student.csv'
  args.dev_out=f"sentiment_results/{timestamp}/dev_out.csv"
  args.test_out=f"sentiment_results/{timestamp}/test_out.csv"
  args.summary_path=f"sentiment_results/sentiment_summaries.jsonl"
  args.plot_path=f"sentiment_results/{timestamp}/training_evolution.png"

  print(f'\n ==== Training Sentiment Classifier on {args.mode.upper()}... ====\n')
  train(args)
  flush_memory()

  print(f'\nEvaluating sentiment analysis on {args.mode.upper()}...')
  test(args)
  flush_memory()

  if not args.keep_model_checkpoint:
    Path(args.filepath).unlink()