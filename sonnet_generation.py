'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW
from evaluation import sonnets_eval

from utils import sync_if_cuda, flush_memory
import time
from datetime import datetime
import json
from pathlib import Path
import gc

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


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for sonnet generation."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask, kv_cache=None):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    output = self.gpt(input_ids, attention_mask, kv_cache)
    last_hidden_states = output["last_hidden_state"]
    logits = self.gpt.hidden_state_to_token(last_hidden_states)

    kv_cache = output["kv_cache"]
    return logits, kv_cache

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate_top_p(self, encoding, temperature=0.7, top_p=0.9, max_length=128): 
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.

    Expects encoding.shape = (batch_size==1, seq_len)
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    kv_cache = None

    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence, kv_cache = self.forward(token_ids, attention_mask, kv_cache)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    token_ids = token_ids[0] # Remove batch dim (1, seq) -> (seq,)
    generated_output = self.tokenizer.decode(token_ids.cpu().numpy().tolist())
    return token_ids, generated_output

  @torch.no_grad()
  def generate_beam(self, encoding, num_beams=5, max_length=128, length_penalty=0.6):
    """
    Beam search autoregressive generation.
    Expects encoding.shape = (batch_size==1, seq_len)
    """
    beams = encoding.to(self.get_device())
    is_finished = [False]
    scores = torch.zeros(1, device=self.get_device())
    attention_mask = torch.ones(beams.shape, dtype=torch.int64).to(self.get_device())
    lengths = torch.tensor([encoding.shape[1]], device=self.get_device(), dtype=torch.int64)
    kv_cache = None

    for _ in range(max_length):

      """ Debugging
      if _ > 100:
        print("==========================")
        print(f"{_} iter BEAMS:")
        print()
        for i in range(beams.shape[0]):
          decoded = self.tokenizer.decode(beams[i])
          print("Decoded beam:")
          print(decoded)
          print()
          print("Cumuulative score:", scores[i])
          print()
          print("Length:", lengths[i])
          print()
          print("Is finished:", is_finished[i])
        print("==========================")
        input("keep debuging")
        """
      
      # Forward pass to get log probs
      logits_sequence, kv_cache = self.forward(beams, attention_mask, kv_cache)
      logits_last_token = logits_sequence[:, -1, :]
      last_token_log_probs = F.log_softmax(logits_last_token, -1) # (Beams, Vocab)

      # Get top probs for each beam
      top_probs, top_idx = torch.topk(last_token_log_probs, num_beams, dim=-1) # Both (Beams, Top B tokens)

      # Select top B from B^2 possibilities
      candidates = []
      for beam in range(top_probs.shape[0]):

        if is_finished[beam]:  # Directly insert finished beams with dummy next-token-postion
          cummulative_prob = scores[beam]
          candidates.append((cummulative_prob, beam, 0))
          continue
              
        for p in range(top_probs.shape[1]): # p keeps track of the place in top_probs, to retrieve top_idx afterwards

          cummulative_prob = scores[beam] + top_probs[beam][p]

          # Keep track of idx from where top_prob was obtained
          candidates.append((cummulative_prob, beam, p))

      def normalize_lengths(cumm_prob, beam_idx, beams_length, is_finished, length_penalty=length_penalty):
        # Normalize counting the token we are about to add
        if is_finished[beam_idx]:
          next_len = beams_length[beam_idx]
        else:
          next_len = beams_length[beam_idx] + 1

        return cumm_prob / next_len ** length_penalty
      
      top_b = sorted(candidates, key=lambda x: normalize_lengths(x[0], x[1], lengths, is_finished), reverse=True)[:num_beams]

      new_beams = []
      new_scores = []
      for score, beam, p in top_b:

        # Retrieve vocab token from saved idx
        if is_finished[beam]:
          token = self.tokenizer.eos_token_id # as padding
        else:
          token = top_idx[beam][p].item()

        new_b = beams[beam].tolist() + [token]
        new_beams.append(new_b)
        new_scores.append(score)

      # Order kv cache to match new beams position
      beams_ids = torch.tensor([b for _, b, _ in top_b], device=self.get_device())
      kv_cache = [(keys[beams_ids], values[beams_ids]) for (keys, values) in kv_cache]

      # Update current beam search state
      beams = torch.tensor(new_beams, device=self.get_device())
      is_finished = [beam[-1].item() == self.tokenizer.eos_token_id for beam in beams]
      scores = torch.tensor(new_scores, device=self.get_device())
      attention_mask = torch.ones(beams.shape, dtype=torch.int64).to(self.get_device())
      lengths = torch.tensor(
        [(b != self.tokenizer.eos_token_id).sum().item() for b in beams],
        device=self.get_device(), dtype=torch.int64
      )

      if all(is_finished):
        break

    # Finally get best beam 
    normalized_scores = scores / lengths.float() ** length_penalty
    best_idx = torch.argmax(normalized_scores)
    best_beam = beams[best_idx]

    # Remove posible padding
    best_beam = best_beam[:lengths[best_idx]]

    # Decode output
    generated_output = self.tokenizer.decode(best_beam.cpu().numpy().tolist())

    return best_beam, generated_output
        

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

  if args.use_gpu:
    torch.cuda.reset_peak_memory_stats()

  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_train)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_dev)
  held_out_labels_dataset = SonnetsDataset(args.held_out_sonnet_dev_labels)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  best_chrf = 0
  args.best_epoch = 0
  epochs_without_improvement = 0

  # Run for the specified number of epochs.
  sync_if_cuda()
  start = time.time()

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits, _ = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    print("Evaluating on dev held out sonnets") ### EVALUATION CODE IS NOT BATCHED SINCE MODEL.GENERATE() ISNT ORIGINALLY BATCHED
    model.eval()
    generated_sonnets = []
    for sonnet_held_out in tqdm(held_out_sonnet_dataset, total=len(held_out_sonnet_dataset)):
      sonnet_id = sonnet_held_out[0]
      encoding = model.tokenizer(sonnet_held_out[1], return_tensors='pt', padding=True, truncation=True).to(device)
      
      if args.generation_method == "top_p":
        output = model.generate_top_p(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      elif args.generation_method == "beam":
        output = model.generate_beam(encoding["input_ids"], num_beams=args.num_beams, length_penalty=args.length_penalty)
      else:
        raise ValueError(f"Unknown generation method: {args.generation_method}")

      generated_sonnets.append((sonnet_id, output[1]))

    
    total_chrf = sonnets_eval(generated_sonnets, held_out_labels_dataset, held_out_sonnet_dataset)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev CHRF :: {total_chrf :.3f}")

    ## Early stopping
    if total_chrf > best_chrf:
      best_chrf = total_chrf
      args.best_epoch = epoch
      save_model(model, optimizer, args, args.filepath)
      epochs_without_improvement = 0 

    else:
      epochs_without_improvement += 1

    if epochs_without_improvement >= args.patience:
      print(f"Early stopping at epoch {epoch}")
      print(f"Best epoch was {args.best_epoch}")
      break

  # Save training time
  sync_if_cuda()
  args.train_time = time.time() - start

  # Save memory usage
  if args.use_gpu:
    args.train_peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    args.train_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9


@torch.no_grad()
def generate_submission_sonnets(args): ### EVALUATION CODE IS NOT BATCHED SINCE MODEL.GENERATE() ISNT ORIGINALLY BATCHED
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  if args.use_gpu:
    torch.cuda.reset_peak_memory_stats()

  saved = torch.load(args.filepath, weights_only=False, map_location=device)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  ## DEV
  dev_dataset = SonnetsDataset(args.held_out_sonnet_dev)
  dev_labels_dataset = SonnetsDataset(args.held_out_sonnet_dev_labels)

  dev_sonnets = []

  sync_if_cuda()
  start = time.time()


  print("Generating dev sonnets")
  for sonnet_held_out in tqdm(dev_dataset, total=len(dev_dataset)):

    # Sonnet generation
    sonnet_id = sonnet_held_out[0]
    encoding = model.tokenizer(sonnet_held_out[1], return_tensors='pt', padding=False, truncation=True).to(device)

    if args.generation_method == "top_p":
      output = model.generate_top_p(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    elif args.generation_method == "beam":
      output = model.generate_beam(encoding["input_ids"], num_beams=args.num_beams, length_penalty=args.length_penalty)
    else:
      raise ValueError(f"Unknown generation method: {args.generation_method}")

    dev_sonnets.append((sonnet_id, output[1]))

  total_chrf = sonnets_eval(dev_sonnets, dev_labels_dataset, dev_dataset)
  print("Dev CHRF: ", total_chrf)


  ## TEST
  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  test_dataset = SonnetsDataset(args.held_out_sonnet_test)

  test_sonnets = []
  print("Generating test sonnets...")
  for batch in tqdm(test_dataset):
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)

    if args.generation_method == "top_p":
      output = model.generate_top_p(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    elif args.generation_method == "beam":
      output = model.generate_beam(encoding["input_ids"], num_beams=args.num_beams, length_penalty=args.length_penalty)
    else:
      raise ValueError(f"Unknown generation method: {args.generation_method}")

    test_sonnets.append((sonnet_id, output[1]))

  sync_if_cuda()
  args.evaluation_time = time.time() - start

  # Save memory usage
  if args.use_gpu:
    args.eval_peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    args.eval_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

  ### Saving
  for path in [args.sonnet_dev_out, args.sonnet_test_out, args.summary_path]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

  # Save dev predictions
  with open(args.sonnet_dev_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in dev_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(f"\n{sonnet[1]}\n")

  # Save test predictions
  with open(args.sonnet_test_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in test_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(f"\n{sonnet[1]}\n")

  with open(args.summary_path, "w") as f:
    data = {"dev_chrf": total_chrf, **vars(args)}
    json.dump(data, f, indent=2)


def get_args():
  parser = argparse.ArgumentParser()

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

  parser.add_argument("--sonnet_train", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_dev", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--held_out_sonnet_dev_labels", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--held_out_sonnet_test", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_dev_out", type=str, default=f"sonnet_results/{timestamp}/generated_sonnets_dev.txt")
  parser.add_argument("--sonnet_test_out", type=str, default=f"sonnet_results/{timestamp}/generated_sonnets_test.txt")
  parser.add_argument("--summary_path", type=str, default=f"sonnet_results/{timestamp}/summary.json")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--generation_method", type=str, help="Generation method for performing sonnet generation.",
                      choices=["beam", "top_p"], default="top_p")
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=0.9)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--num_beams", type=int, help="Number of beams for beam search generation.", default=5)
  parser.add_argument("--length_penalty", type=float, help="Length penalty for beam search scoring.", default=0.6)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  
  parser.add_argument("--generate_only", action="store_true", help="If applied, program skips training and loads saved weights.")

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
  args.filepath = f'checkpoints/{args.model_size}-sonnet.pt'  # Model save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  if not args.generate_only:
    train(args)
    flush_memory()
  generate_submission_sonnets(args)