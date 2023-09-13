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

    # kill_parameters_splits = [[["C", "S"], ["M", "S"], ["S", "Z"], ["M", "Z"]],
    #                           [["M", "C"], ["M", "Z"], ["C", "S"], ["C", "Z"]]]

    kill_parameters_splits = [[["C", "S"], ["M", "S"], ["C", "Z"], ["M", "Z"]],
                              [["M", "S"], ["M", "Z"], ["C", "S"], ["C", "Z"]]]

    params_list = [{"type": "Size", "name": "<Z>"},
                   {"type": "Color", "name": "<C>"},
                   {"type": "Material", "name": "<M>"},
                   {"type": "Shape", "name": "<S>"},
                   {"type": "Size", "name": "<Z2>"},
                   {"type": "Color", "name": "<C2>"},
                   {"type": "Material", "name": "<M2>"},
                   {"type": "Shape", "name": "<S2>"}]

    for dir_split, kill_parameters in zip(dirs_splits, kill_parameters_splits):
        list_dict = []

        kill_ = kill_parameters[0]
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
                          "constraints": [{"params": ["<Z>"], "type": "NULL"},
                                          {"params": ["<Z2>"], "type": "NULL"},
                                          {"params": ["<%s>" % kill_[0]], "type": "NULL"},
                                          {"params": ["<%s2>" % kill_[0]], "type": "NULL"},
                                          {"params": ["<%s>" % kill_[1]], "type": "NULL"},
                                          {"params": ["<%s2>" % kill_[1]], "type": "NULL"},
                                          {"params": [1, 3], "type": "OUT_NEQ"}]})

        kill_ = kill_parameters[1]
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
                          "constraints": [{"params": ["<C>"], "type": "NULL"},
                                          {"params": ["<C2>"], "type": "NULL"},
                                          {"params": ["<%s>" % kill_[0]], "type": "NULL"},
                                          {"params": ["<%s2>" % kill_[0]], "type": "NULL"},
                                          {"params": ["<%s>" % kill_[1]], "type": "NULL"},
                                          {"params": ["<%s2>" % kill_[1]], "type": "NULL"},
                                          {"params": [1, 3], "type": "OUT_NEQ"}]})

        kill_ = kill_parameters[2]
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
                          "constraints": [{"params": ["<M>"], "type": "NULL"},
                                          {"params": ["<M2>"], "type": "NULL"},
                                          {"params": ["<%s>" % kill_[0]], "type": "NULL"},
                                          {"params": ["<%s2>" % kill_[0]], "type": "NULL"},
                                          {"params": ["<%s>" % kill_[1]], "type": "NULL"},
                                          {"params": ["<%s2>" % kill_[1]], "type": "NULL"},
                                          {"params": [1, 3], "type": "OUT_NEQ"}]})

        kill_ = kill_parameters[3]
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
                          "constraints": [{"params": ["<S>"], "type": "NULL"},
                                          {"params": ["<S2>"], "type": "NULL"},
                                          {"params": ["<%s>" % kill_[0]], "type": "NULL"},
                                          {"params": ["<%s2>" % kill_[0]], "type": "NULL"},
                                          {"params": ["<%s>" % kill_[1]], "type": "NULL"},
                                          {"params": ["<%s2>" % kill_[1]], "type": "NULL"},
                                          {"params": [1, 3], "type": "OUT_NEQ"}]})

        os.makedirs(join(info_path, dir_split))
        with open(join(info_path, dir_split, 'comparison.json'), 'w') as f:
            json.dump(list_dict, f)

