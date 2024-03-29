from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np


def batch_apk(logits, pos_mask, k):
  """Computes Average Precision at cut-off k.
  Args:
    logits: [B, M]. Predicted relevance of each candidate.
    pos_mask: [B, M]. Ground truth binary relevance of each candidate.
  Returns:
    ap_score: [B]. Average precision of the induced ranking.
  """
  ranks = np.argsort(logits, axis=1)[:, ::-1]  # for example [1, 0, 3, 2]
  actual = np.zeros_like(logits)
  predicted = np.zeros_like(logits)
  for ii in range(logits.shape[0]):
    actual[ii] = pos_mask[ii, ranks[ii]]
    predicted[ii] = logits[ii, ranks[ii]]
  num_items = np.minimum(k, predicted.shape[1])
  hits = actual
  hits = np.cumsum(hits, axis=1)
  mask = (np.expand_dims(np.arange(actual.shape[1]), 0) < np.expand_dims(
      num_items, 1)).astype(np.float32)
  hits *= mask
  hits *= actual
  denom = np.arange(actual.shape[1]) + 1.0
  score = hits / denom
  score = score.sum(axis=1)
  num_relevant_at_k = np.maximum(np.minimum(k, actual.sum(axis=1)), 1.0)
  ap_score = score / num_relevant_at_k
  return ap_score


# def batch_apk(logits, pos_mask, k):
#   ap_score = np.array(
#       [apk(logits[ii], pos_mask[ii], k) for ii in range(logits.shape[0])])
#   # print('measure', ap_score)
#   return ap_score


def apk(logits, pos_mask, k):
  """Computes Average Precision at cut-off k.
  Args:
    logits: [M]. Predicted relevance of each candidate.
    pos_mask: [M]. Ground truth binary relevance of each candidate.
  Returns:
    ap_score: []. Average precision of the induced ranking.
  """
  ranks = np.argsort(logits)[::-1]  # for example [1, 0, 3, 2]
  # print('ranks', ranks)
  actual = np.array(pos_mask)[ranks]
  # print('sorted', actual)
  # print('logits', logits)
  predicted = np.array(logits)[ranks]
  #log.info("actual: {}".format(actual))
  #log.info("predicted: {}".format(predicted))
  score = 0.0
  num_hits = 0.0
  for ii in range(min(k, len(predicted))):
    if actual[ii]:
      num_hits += 1.0
      score += num_hits / (ii + 1.0)
  num_relevant_at_k = max(min(k, len(np.where(actual == 1)[0])), 1.0)
  ap_score = score / num_relevant_at_k
  return ap_score

'''
def ap(logits, pos_mask):
  """Computes Average Precision.
  Args:
    logits: [M]. Predicted relevance of each candidate.
    pos_mask: [M]. Ground truth binary relevance of each candidate.
  Returns:
    ap_score: []. Average precision of the induced ranking.
  """
  rank = np.argsort(logits)[::-1]
  actual = np.array(pos_mask)[rank]
  predicted = np.array(logits)[rank]
  num_pos = pos_mask.sum()
  score = 0.0
  num_hits = 0.0
  for ii in range(len(predicted)):
    if actual[ii]:
      num_hits += 1.0
      score += num_hits / (ii + 1.0)
  num_relevant_at_k = max(np.where(actual == 1)[0], 1.0)
  return ap_score / num_relevant_at_k
'''