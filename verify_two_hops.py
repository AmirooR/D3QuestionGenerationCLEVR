import argparse
import os
import json
import numpy as np
import h5py

parser = argparse.ArgumentParser()

parser.add_argument('--ood_questions', type=str,
    default="data/two_hop_datasets/2Hop/test_ood/valB_questions.h5",
    help="Path to OOD questions h5 file.")

parser.add_argument('--vocab', type=str,
    default="vocab.json",
    help="Path to vocab file.")

ind_combinations = {
  'count': {'color','size','left','front'},
  'exist': {'material','shape','right','behind'},
}

swap_combinations = {
  'exist': {'color','size','left','front'},
  'count': {'material','shape','right','behind'},
}

def inv_dict(d): return {v:k for k,v in d.items()}

class TwoHopVerifier():
  def __init__(self, vocab, programs):
    self.vocab = vocab
    self.programs = programs
    self.id2token = inv_dict(vocab['program_token_to_idx'])

  def get_question_type(self, program):
    return self.id2token[program[1]]

  def get_question_attrs(self, program):
    #returns a set
    tokens = [self.id2token[x] for x in program]
    attrs = set()
    for token in tokens:
      if token.startswith('relate'):
        attr = token[token.find('[')+1:-1]
        attrs.add(attr)
      elif token.startswith('filter'):
        attr = token[token.find('_')+1:token.find('[')]
        attrs.add(attr)
    return attrs

  def is_same_or_swap(self, program):
    qtype = self.get_question_type(program)
    qattrs = self.get_question_attrs(program)
    return qattrs.issubset(ind_combinations[qtype]) or qattrs.issubset(swap_combinations[qtype])

  def verify_all(self):
    mask = []
    for program in self.programs:
      no_ood = self.is_same_or_swap(program)
      mask.append(no_ood)
    return mask

def main(args):
  programs = h5py.File(args.ood_questions,'r')['programs'][:]
  vocab = json.load(open(args.vocab,'r'))
  verifier = TwoHopVerifier(vocab, programs)
  mask = verifier.verify_all()

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

