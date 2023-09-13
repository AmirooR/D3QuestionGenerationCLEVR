import json
import itertools
import os
import copy

def add_null_constraint(template, attr, depth):
  for i in range(depth):
    s = '' if i == 0 else i+1
    constraint = {"params": [f"<{attr}{s}>"], "type": "NULL"}
    template['constraints'].append(constraint)

def add_oneof_constraint(template, attrs, depth):
  for i in range(depth):
    s = '' if i == 0 else i+1
    constraint = {"params":[f"<R{s}>", attrs], "type": "VAL_ONEOF"}
    template['constraints'].append(constraint)

def main(two_hop_path, output_dir, depth=3, output_name='two_hop'):
  os.makedirs(os.path.join(output_dir,'train'), exist_ok=True)
  os.makedirs(os.path.join(output_dir,'test'), exist_ok=True)
  two_hop = json.load(open(two_hop_path, 'r'))
  two_hop = two_hop[:2] # 0: count, 1: exist
  attrs = {'M','C','S','Z'}
  rels = {'left','right','front','behind'}
  countInD = {'C','Z','left','front'}
  existInD = {'M','S','right','behind'}
  ind_two_hop = copy.deepcopy(two_hop)

  # count
  add_null_constraint(ind_two_hop[0], attr='M', depth=depth)
  add_null_constraint(ind_two_hop[0], attr='S', depth=depth)
  add_oneof_constraint(ind_two_hop[0], ['left','front'], depth=depth-1)

  # exist
  add_null_constraint(ind_two_hop[1], attr='C', depth=depth)
  add_null_constraint(ind_two_hop[1], attr='Z', depth=depth)
  add_oneof_constraint(ind_two_hop[1], ['right','behind'], depth=depth-1)

  with open(os.path.join(output_dir, 'train', f'{output_name}.json'), 'w') as f:
    json.dump(ind_two_hop, f)

  ood_two_hop = []

  for a1,a2 in itertools.combinations(attrs, 2):
    for r1,r2 in itertools.combinations(rels, 2):
      if len({a1,a2,r1,r2}.intersection(countInD)) == 2 and len({a1,a2,r1,r2}.intersection(existInD)) == 2:
        template = {"constraints":[]}
        for attr in attrs - {a1,a2}:
          add_null_constraint(template, attr=attr, depth=depth)
        add_oneof_constraint(template, [r1,r2], depth=depth-1)

        # count
        ood_template = copy.deepcopy(two_hop[0])
        ood_template['constraints'] = template['constraints']
        ood_two_hop.append(ood_template)

        # exist
        ood_template = copy.deepcopy(two_hop[1])
        ood_template['constraints'] = template['constraints']
        ood_two_hop.append(ood_template)

  with open(os.path.join(output_dir, 'test', f'{output_name}.json'), 'w') as f:
    json.dump(ood_two_hop, f)


if __name__ == '__main__':
  main('CLEVR_1.0_templates/two_hop.json', 'two_hop_qtype_templates', depth=3, output_name='two_hop')
  main('CLEVR_1.0_templates/three_hop.json', 'two_hop_qtype_templates', depth=4, output_name='three_hop')
