import argparse
import os
import sys
import json
import random
import numpy as np
from copy import deepcopy

parser = argparse.ArgumentParser()
#parser.add_argument('--question_json_files', nargs='+', type=str, required=True)
parser.add_argument('--wide_json_file', type=str, required=True)
parser.add_argument('--catalog_json_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--total_samples', type=int, default=None)
parser.add_argument('--input_vocab_json', type=str, default="vocab.json")
parser.add_argument('--seed', type=int, default=1357)


def read_json(path):
  return json.load(open(path, 'r'))

def main(args):
  random.seed(args.seed)
  catalog = read_json(args.catalog_json_file)
  wide_questions = read_json(args.wide_json_file)
  dataset_lens = [info['len'] for info in catalog['training_set_info']]
  cumsums = np.cumsum([0] + dataset_lens)
  starts = cumsums[:-1]
  ends = cumsums[1:]
  if args.total_samples is None:
    total = len(wide_questions['questions'])
  else:
    total = args.total_samples

  for training_set in catalog['training_sets']:
    print(f"Creating dataset {training_set['name']}")
    assert(len(training_set['fractions']) == len(starts))
    questions = []
    for frac,start,end in zip(training_set['fractions'], starts, ends):
      num_to_sample = int(frac*total)
      questions_to_sample = wide_questions['questions'][start:end]
      questions.extend(random.sample(questions_to_sample, num_to_sample))

    questions = deepcopy(questions)
    for i, q in enumerate(questions):
      q['question_index'] = i
    d = {'info': wide_questions['info'], 'questions': questions}
    dataset_root = os.path.join(args.output_dir, training_set['name'])
    os.makedirs(dataset_root, exist_ok=True)
    questions_json = os.path.join(dataset_root, 'train_questions.json')
    questions_h5 = os.path.join(dataset_root, 'train_questions.h5')
    with open(questions_json, 'w') as f:
      json.dump(d, f)

    from easydict import EasyDict

    flags = EasyDict({
      'input_questions_json': questions_json,
      'output_h5_file': questions_h5,
      'input_vocab_json': args.input_vocab_json,
      'mode': 'prefix',
      'expand_vocab': 0,
      'unk_threshold': 1,
      'encode_unk': 0,
      'output_vocab_json': '',
    })

    sys.path.append('clevr-iep')
    from scripts import preprocess_questions
    preprocess_questions.main(flags)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

