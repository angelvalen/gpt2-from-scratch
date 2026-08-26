# !/usr/bin/env python3

"""
Evaluation code for Quora paraphrase detection.

model_eval_paraphrase is suitable for the dev (and train) dataloaders where the label information is available.
model_test_paraphrase is suitable for the test dataloader where label information is not available.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  SonnetsDataset,
)


TQDM_DISABLE = False


@torch.no_grad()
def model_eval_sentiment(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    b_labels = b_labels.flatten()
    y_true.extend(b_labels)
    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sents, sent_ids


@torch.no_grad()
def model_test_sentiment(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
      'labels'].flatten()

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_true.extend(labels)
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)


def sonnets_eval(generated, gold, held_out):

    # Ensure generated, gold and held out sonnets match size
    assert len(generated) == len(gold) == len(held_out)

    # Assert generated and gold contain held out part
    assert generated[0][1][:len(held_out[0][1])] == held_out[0][1]
    assert gold[0][1][:len(held_out[0][1])] == held_out[0][1]

    # Cut held out part
    hypothesis = [x[1][len(y[1]):] for x, y in zip(generated, held_out)]
    reference = [x[1][len(y[1]):] for x, y in zip(gold, held_out)]

    chrf = CHRF()
    chrf_score = chrf.corpus_score(hypothesis, [reference])

    return float(chrf_score.score)


# Previous implementation i did, unused
from collections import Counter
import warnings
def compute_chrf(held_out_reference, hypothesis, reference, beta=1):
  """Computes CHRF score (https://aclanthology.org/anthology-files/pdf/W/W15/W15-3049.pdf)
  for the predicted continuation against a gold reference"""

  # Remove initial reference from chrf score
  hypothesis = hypothesis[len(held_out_reference):]
  reference = reference[len(held_out_reference):]

  if not hypothesis:
    return 0.0

  precision = 0.
  recall = 0.
  for n in range(1, 7):

    hyp_n_grams = [hypothesis[i:i+n] for i in range(len(hypothesis) - n + 1)]
    ref_n_grams = [reference[i:i+n] for i in range(len(reference) - n + 1)]

    if not hyp_n_grams or not ref_n_grams:
            msg = f"""Couldnt extract {n}-grams for hypothesis: 
            {hypothesis} 
            and reference: 
            {reference}."""
            warnings.warn(msg)
            continue
    
    hyp_counts = Counter(hyp_n_grams)
    ref_counts = Counter(ref_n_grams)

    matches = sum(min(hyp_counts[gram], ref_counts[gram]) for gram in hyp_counts)
    precision += matches / len(hyp_n_grams)
    recall += matches / len(ref_n_grams)

  # Return 0 if no overlap at all
  if precision + recall == 0:
    return 0.0

  precision /= 6
  recall /= 6
  chrf = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

  return chrf