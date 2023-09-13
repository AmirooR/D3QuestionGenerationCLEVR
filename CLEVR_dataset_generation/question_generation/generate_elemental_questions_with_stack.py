import numpy as np
import os
import json
from os.path import join
import h5py


def invert_dict(dct):
    new_dct = {}
    for k_, v_ in dct.items():
        new_dct[v_] = k_
    return new_dct


def sum_next(array):
    array = array.astype(np.int32)
    sum_ = np.array([array[i]+array[i+1] for i in range(array.size-1)])
    idxs = [[i, i+1] for i in range(array.size-1)]
    return sum_, idxs


def force_program_structure(questions_folder,
                            vocab_json,
                            allow_no_stack=False,
                            force_and=False):
    """
    You need to provide a dataset where features and vocabulary
    and questions with stacks are already present.
    The questions will be in the h5 file and contained in a folder

    Q1: <S> object equal_size of the <M> object?
    Q2: <C><S> object equal_size of the <M> object?
    Q3: <C><S> object equal_size of the <M><Z> object?

    if define_stack == 1:
        Q1 False
        Q2 and Q3 True because we have 1 stack

    elif define_stack == 2:
        Q1 and Q2 False
        Q3 True because we have two stacks
    """

    vocab = json.load(open(vocab_json, 'rb'))
    id_filters = [i_ for k_, i_ in vocab['program_token_to_idx'].items() if k_.startswith('filter')]
    dir_different_struct = join(questions_folder, 'force_structure')
    os.makedirs(dir_different_struct, exist_ok=True)

    for q_file in [q_ for q_ in os.listdir(questions_folder) if q_.endswith('.h5')]:
        q_h5 = h5py.File(join(questions_folder, q_file), 'r')
        programs = q_h5['programs'][:]

        bm = np.zeros(programs.shape, dtype=bool)
        for id_f in id_filters:  # check where the filters sub-tasks appear
            bm = np.logical_or(programs == id_f, bm)
        # which programs do have the stack of multiple filters

        if allow_no_stack:
            if not force_and:
                print('Questions are already available!')
                return
            else:
                id_stack = np.arange(programs.shape[0])
                id_no_need_and = list(np.argwhere(np.sum(bm, axis=1) < 2).squeeze())

        else:
            # in case of non elemental questions
            # idx_double_concat = []
            # for id_sample, bm_single_program in enumerate(bm):
            #     sum_bm, indexes_concat = sum_next(bm_single_program)
            #     print(sum_bm)
            #     if np.sum(sum_bm == 2) >= define_stack:
            #         # we expect programs with at least 2 or more concatenated filters
            #         # for at least a group of objects -- it works with the elemental questions as well
            #         idx_double_concat.append(id_sample)
            # id_stack = np.array(idx_double_concat, dtype=np.int32)
            id_stack = np.argwhere(np.sum(bm, axis=1) >= 2).squeeze()  # if they appear more than once
            id_no_need_and = []

        # generate h5 file
        if allow_no_stack:
            name_file = 'and_'
        else:
            name_file = 'stack_and_' if force_and else 'stack'
        f = h5py.File(join(dir_different_struct, name_file + q_file), 'w')
        if not force_and:
            for k_ in q_h5.keys():
                tmp_array = q_h5[k_][id_stack]
                dset = f.create_dataset(k_,
                                        shape=tmp_array.shape,
                                        dtype=np.int32,
                                        chunks=True)
                dset[:] = tmp_array

        else:
            new_programs = []
            len_pg = np.zeros(id_stack.size, dtype=np.int32)
            for count_id, id_p in enumerate(id_stack):
                if allow_no_stack and id_p in id_no_need_and:
                    new_programs.append(list(programs[id_p]))

                else:
                    new_pg = []
                    for q_ in programs[id_p]:
                        if not (q_ in id_filters):
                            new_pg.append(q_)
                        else:
                            filter_ = q_
                            if filter_ != q_h5['programs'][id_p][bm[id_p]][-1]:
                                new_pg.append(vocab['program_token_to_idx']['intersect'])
                            new_pg.append(filter_)
                            if filter_ != q_h5['programs'][id_p][bm[id_p]][-1]:
                                new_pg.append(vocab['program_token_to_idx']['scene'])
                    len_pg[count_id] = len(new_pg)
                    new_programs.append(new_pg)
            max_len = np.max(len_pg)
            print(vocab['program_token_to_idx']['<NULL>'])
            print(id_stack.size, max_len)
            new_program_matrix = vocab['program_token_to_idx']['<NULL>'] * np.ones((id_stack.size, max_len), dtype=np.int32)
            for id_pg_, pg_ in enumerate(new_programs):
                tmp_len = len(pg_)
                new_program_matrix[id_pg_, :tmp_len] = pg_

            for k_ in q_h5.keys():
                if k_ != 'programs':
                    tmp_array = q_h5[k_][id_stack]
                else:
                    tmp_array = new_program_matrix
                dset = f.create_dataset(k_,
                                        shape=tmp_array.shape,
                                        dtype=np.int32,
                                        chunks=True)
                dset[:] = tmp_array
