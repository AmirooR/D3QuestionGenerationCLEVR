import json
from os.path import join
import os


def compute_combinations(elements=['C', 'M', 'S', 'Z'], exclude='Z'):
    copy_elements = elements.copy()
    copy_elements.remove(exclude)

    list_comb = []
    for id1, e1 in enumerate(copy_elements):
        for e2 in copy_elements[id1 + 1:]:
            list_comb.append([e1, e2])
    return list_comb


def run(info_path):

    dirs_splits = ['train', 'test']

    #                             size        color      material    shape 
    #kill_parameters_splits = [[["C", "S"], ["M", "S"], ["C", "Z"], ["M", "Z"]],
    #                          [["M", "S"], ["M", "Z"], ["C", "S"], ["C", "Z"]]]

    allowed_parameters_splits = [[["M", "S"], ["M", "Z"], ["C", "S"], ["C", "Z"]], # training is not important
                               [["C", "S"], ["M", "S"], ["C", "Z"], ["M", "Z"]]]


    params_list = [{"type": "Size", "name": "<Z>"},
                   {"type": "Color", "name": "<C>"},
                   {"type": "Material", "name": "<M>"},
                   {"type": "Shape", "name": "<S>"},
                   {"type": "Size", "name": "<Z2>"},
                   {"type": "Color", "name": "<C2>"},
                   {"type": "Material", "name": "<M2>"},
                   {"type": "Shape", "name": "<S2>"}]
    attrs = ['Z','S','C','M']


    def get_constraints(allowed, i):
      constraints = []
      visited = False
      for param in attrs:
        if param in allowed:
          if not visited:
            constraints.append({'params': [f"<{allowed[i]}>"], "type": "NULL"})
            constraints.append({'params': [f"<{allowed[1-i]}2>"], "type": "NULL"})
            visited = True
        else:
          constraints.append({'params': [f"<{param}>"], "type": "NULL"})
          constraints.append({'params': [f"<{param}2>"], "type": "NULL"})

      return constraints

    for dir_split, allowed_parameters in zip(dirs_splits, allowed_parameters_splits):
        list_dict = []

        allowed = allowed_parameters[0]
        for i in range(2):
          constraints = get_constraints(allowed, i)
          constraints.append({"params": [1, 3], "type": "OUT_NEQ"})
          list_dict.append({"text": ["Do the <Z> <C> <M> <S> and the <Z2> <C2> <M2> <S2> have the same size?",
                                     "Is the size of the <Z> <C> <M> <S> the same as the <Z2> <C2> <M2> <S2>?",
                                     "Is the <Z> <C> <M> <S> the same size as the <Z2> <C2> <M2> <S2>?",
                                     "Does the <Z> <C> <M> <S> have the same size as the <Z2> <C2> <M2> <S2>?"],
                            "nodes": [{"inputs": [], "type": "scene"},
                                      {"side_inputs": ["<Z>", "<C>", "<M>", "<S>"], "inputs": [0], "type": "filter_unique"},
                                      {"inputs": [], "type": "scene"},
                                      {"side_inputs": ["<Z2>", "<C2>", "<M2>", "<S2>"], "inputs": [2],
                                       "type": "filter_unique"},
                                      {"inputs": [1], "type": "query_size"}, {"inputs": [3], "type": "query_size"},
                                      {"inputs": [4, 5], "type": "equal_size"}],
                            "params": params_list,
                            "constraints": constraints})

        allowed = allowed_parameters[1]
        for i in range(2):
          constraints = get_constraints(allowed, i)
          constraints.append({"params": [1, 3], "type": "OUT_NEQ"})

          list_dict.append({"text": ["Do the <Z> <C> <M> <S> and the <Z2> <C2> <M2> <S2> have the same color?",
                                     "Is the color of the <Z> <C> <M> <S> the same as the <Z2> <C2> <M2> <S2>?",
                                     "Does the <Z> <C> <M> <S> have the same color as the <Z2> <C2> <M2> <S2>?",
                                     "Is the <Z> <C> <M> <S> the same color as the <Z2> <C2> <M2> <S2>?"],
                            "nodes": [{"inputs": [], "type": "scene"}, {"side_inputs": ["<Z>", "<C>", "<M>", "<S>"],
                                                                        "inputs": [0], "type": "filter_unique"},
                                      {"inputs": [], "type": "scene"},
                                      {"side_inputs": ["<Z2>", "<C2>", "<M2>", "<S2>"], "inputs": [2],
                                       "type": "filter_unique"},
                                      {"inputs": [1], "type": "query_color"}, {"inputs": [3], "type": "query_color"},
                                      {"inputs": [4, 5], "type": "equal_color"}],
                            "params": params_list,
                            "constraints": constraints})

        allowed = allowed_parameters[2]
        for i in range(2):
          constraints = get_constraints(allowed, i)
          constraints.append({"params": [1, 4], "type": "OUT_NEQ"})

          list_dict.append({"text": ["Do the <Z> <C> <M> <S> and the <Z2> <C2> <M2> <S2> have the same material?",
                                     "Are the <Z> <C> <M> <S> and the <Z2> <C2> <M2> <S2> made of the same material?",
                                     "Is the material of the <Z> <C> <M> <S> the same as the <Z2> <C2> <M2> <S2>?",
                                     "Does the <Z> <C> <M> <S> have the same material as the <Z2> <C2> <M2> <S2>?",
                                     "Is the <Z> <C> <M> <S> made of the same material as the <Z2> <C2> <M2> <S2>?"],
                            "nodes": [{"inputs": [], "type": "scene"}, {"side_inputs": ["<Z>", "<C>", "<M>", "<S>"],
                                                                        "inputs": [0], "type": "filter_unique"},
                                      {"inputs": [1], "type": "query_material"},
                                      {"inputs": [], "type": "scene"}, {"side_inputs": ["<Z2>", "<C2>", "<M2>", "<S2>"],
                                                                        "inputs": [3], "type": "filter_unique"},
                                      {"inputs": [4], "type": "query_material"},
                                      {"inputs": [2, 5], "type": "equal_material"}],
                            "params": params_list,
                            "constraints": constraints})

        allowed = allowed_parameters[3]
        for i in range(2):
          constraints = get_constraints(allowed, i)
          constraints.append({"params": [1, 4], "type": "OUT_NEQ"})

          list_dict.append({"text": ["Do the <Z> <C> <M> <S> and the <Z2> <C2> <M2> <S2> have the same shape?",
                                     " Does the <Z> <C> <M> <S> have the same shape as the <Z2> <C2> <M2> <S2>?",
                                     "Is the shape of the <Z> <C> <M> <S> the same as the <Z2> <C2> <M2> <S2>?",
                                     "Is the <Z> <C> <M> <S> the same shape as the <Z2> <C2> <M2> <S2>?"],
                            "nodes": [{"inputs": [], "type": "scene"},
                                      {"side_inputs": ["<Z>", "<C>", "<M>", "<S>"], "inputs": [0], "type": "filter_unique"},
                                      {"inputs": [1], "type": "query_shape"}, {"inputs": [], "type": "scene"},
                                      {"side_inputs": ["<Z2>", "<C2>", "<M2>", "<S2>"], "inputs": [3],
                                       "type": "filter_unique"},
                                      {"inputs": [4], "type": "query_shape"},
                                      {"inputs": [2, 5], "type": "equal_shape"}],
                            "params": params_list,
                            "constraints": constraints})

        os.makedirs(join(info_path, dir_split))
        with open(join(info_path, dir_split, 'comp2attrs.json'), 'w') as f:
            json.dump(list_dict, f)


if __name__ == '__main__':
  run('comp2attrs')

