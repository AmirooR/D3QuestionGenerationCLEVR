import argparse
import os
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument('--question_json_files', nargs='+', type=str, required=True)
parser.add_argument('--output_json_file', type=str, required=True)
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--seed', type=int, default=1357)


def main(args):
  random.seed(args.seed)
  questions = []
  for f in args.question_json_files:
    d = json.load(open(f,'r'))
    if args.num_samples is None:
      questions.extend(d['questions'])
    else:
      questions.extend(random.sample(d['questions'], args.num_samples))
    info = d['info']

  for i, q in enumerate(questions):
    q['question_index'] = i

  d = {'info': info, 'questions': questions}
  with open(args.output_json_file, 'w') as f:
    json.dump(d, f)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)


