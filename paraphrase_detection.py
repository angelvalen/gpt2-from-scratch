'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

from utils import sync_if_cuda
import time 
from datetime import datetime
import json
from pathlib import Path

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


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
    self.paraphrase_dropout = nn.Dropout(args.paraphrase_dropout_prob)

    # Choos to fine-tune full model or just last layer
    assert args.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if args.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif args.fine_tune_mode == 'full-model':
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    last_hidden = self.gpt(input_ids, attention_mask)["last_token"]
    last_droped = self.paraphrase_dropout(last_hidden)
    logits = self.paraphrase_detection_head(last_droped)
    return logits


def save_model(model, optimizer, args, filepath):

  Path(filepath).parent.mkdir(parents=True, exist_ok=True)

  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  if args.small_datasets:
    para_train_data = para_train_data[:len(para_train_data) // 10]
    para_dev_data = para_dev_data[:len(para_dev_data) // 10]

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
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
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
    
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

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

  # Save total training time
  sync_if_cuda()
  args.train_time = time.time() - start

  # Save memory usage
  if args.use_gpu:
      args.train_peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
      args.train_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath, weights_only=False)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  if args.small_datasets:
    para_test_data = para_test_data[:len(para_test_data) // 10]
    para_dev_data = para_dev_data[:len(para_dev_data) // 10]

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)
  
  sync_if_cuda()
  start = time.time()

  if args.use_gpu:
    torch.cuda.reset_peak_memory_stats()

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  sync_if_cuda()
  args.evaluation_time = time.time() - start

  if args.use_gpu:
    args.eval_peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    args.eval_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

  

  ## Save
  for path in [args.para_dev_out, args.para_test_out, args.summary_path]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.summary_path, "w") as f:
    data = {"dev_accuracy": dev_para_acc, **vars(args)}
    json.dump(data, f, indent=2)


def get_args():
  parser = argparse.ArgumentParser()

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default=f"paraphrase_results/{timestamp}/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default=f"paraphrase_results/{timestamp}/para-test-output.csv")
  parser.add_argument("--summary_path", type=str, default=f"paraphrase_results/{timestamp}/summary.json")

  parser.add_argument("--small_datasets", action="store_true",
                       help="If selected, cuts train, dev and test datasets to be a tenth of their lengths")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--paraphrase_dropout_prob", type=float, default=0.1)

  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  
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
  args.filepath = f'checkpoints/{args.model_size}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
