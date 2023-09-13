import argparse
import os
import sys
import json
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--unbiased_questions', type=str, required=True,
    help="path to unbiased questions json file")
parser.add_argument('--biased_questions', type=str, required=True,
    help="path to biased questions json file")
parser.add_argument('--output_questions', type=str, required=True,
    help="path to output json question file")
parser.add_argument('--fraction', type=float, required=False, default=None,
    help="fraction of biased  dataset to change. should be in range (0,1)." + \
    " Only one of fraction and num_to_change should be set.")
parser.add_argument('--num_to_change', type=int, required=False, default=None,
    help="Number of questions to change in biased questions." + \
        " Only one of fraction and num_to_change should be set."
    )
parser.add_argument('--seed', type=int, required=False, default=1357,
    help="random seed")


def main(args):
  random.seed(args.seed)
  assert((args.fraction is None) ^ (args.num_to_change is None)), f'Only one of fraction ({args.fraction})' + \
  f'and num_to_change ({args.num_to_change}) should be set.'

  with open(args.unbiased_questions, 'r') as f:
    unbiased = json.load(f)
  with open(args.biased_questions, 'r') as f:
    biased = json.load(f)
  biased_qs = biased['questions']
  unbiased_qs = unbiased['questions']
  if args.fraction:
    assert(0 <= args.fraction <= 1), "fraction should be between 0 and 1"
    num_to_change = int(args.fraction * len(biased_qs))
  else:
    num_to_change = args.num_to_change

  assert(num_to_change <= len(unbiased_qs)), f"{num_to_change} is greater than number of unbiased questions {len(unbiased_qs)}"
  assert(num_to_change <= len(biased_qs)), f"{num_to_change} is greater than number of biased questions {len(biased_qs)}"

  # We do simply by substituting some questions from unbiased to biased dataset
  # There can be other strategies (like trying to use the same images)
  # But I ignore it for now!
  selected_unbiased_indices = random.sample(list(range(len(unbiased_qs))), num_to_change)
  selected_biased_indices = random.sample(list(range(len(biased_qs))), num_to_change)

  for i in range(num_to_change):
    unbiased_q = unbiased_qs[selected_unbiased_indices[i]]
    biased_q = biased_qs[selected_biased_indices[i]]
    # Assign some keys (not sure if these are necessary)
    unbiased_q['split'] = biased_q['split']
    unbiased_q['question_index'] = biased_q['question_index']
    if i < 5:
      print(f"[{biased_q['question_index']}]: {unbiased_q['question']} ---> {biased_q['question']}")
    biased_qs[selected_biased_indices[i]] = unbiased_q

  with open(args.output_questions, 'w') as f:
    json.dump(biased, f)

  print(f"Changed {num_to_change} questions")


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

