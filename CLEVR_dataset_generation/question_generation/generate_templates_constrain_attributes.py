import argparse
import json
import os
import itertools  # random, shutil
from os.path import join
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--folder_CLEVR_templates',
                    default='CLEVR_dataset_generation/question_generation/CLEVR_1.0_templates')
parser.add_argument('--keep_templates', nargs='+', default=[])
parser.add_argument('--question_bias', action='store_true')
parser.add_argument('--bias_strength', type=str, choices=['ultrastrict', 'strict', None], default=None)
parser.add_argument('--exclude_questions_with', nargs='+', default=[])
parser.add_argument('--stack', action='store_true')
# parser.add_argument('--tree_structure', action='store_true')
parser.add_argument('--folder_output_templates',
                    default='CLEVR_dataset_generation/question_generation/folder_templates')


def flatten(lst):
    """Flatten a list."""
    return [y for l in lst for y in flatten(l)] if isinstance(lst, (list, np.ndarray)) else [lst]


def grouping_attributes(template):
    """
    Given a template, we consider its parameters (those not set to NULL).
    We remove the relations, as those do not appear concatenated in the program.
    We group the attributes which are not NULL depending if referring to a first group of object
    <C><M><Z><S>, a second group <C2><M2><Z2><S2> and so on.
    """
    constraints = flatten([constraints['params'] for constraints in template['constraints']
                           if constraints['type'] == 'NULL'])  # only those that impose an attribute to be null
    name_params = [param['name'] for param in template['params']]
    for c_ in constraints:  # we remove those params from the one over which we impose the split
        name_params.remove(c_)
    for n_ in name_params:
        if n_.startswith('<R'):  # we do the same for spatial relations
            name_params.remove(n_)
    print('name params: ', name_params)

    parsed = [p_.split('<')[-1].split('>')[0] for p_ in name_params]  # parsing
    split_into_groups = np.array([p_[1:] for p_ in parsed])  # "", "2", "3", etc identifiers
    group_by = {k_ + 1: [] for k_ in range(np.size(np.unique(np.array(split_into_groups))))}

    for k_, id_group in zip(group_by.keys(), np.unique(split_into_groups)):
        group_by[k_] = list(np.array(name_params)[id_group == split_into_groups])  # attributes per group
    return group_by


def impose_strict_bias_attributes(template, fewer_at_train=True, extra_bias=False, same_subtask_input=False):
    """ This way of imposing bias works if we want to reduce the attributes in the stack
    e.g. from How many <Z> <C> <M> <S> are there?
    we want to exclude two out for the four attributes.

    We first consider the original constraints of the question.
    Then we divide the attributes depending on which group of objects they identify
    <Z> <C> <M> <S>, <Z2> <C2> <M2> <S2>, <Z3> <C3> <M3> <S3>, <Z4> <C4> <M4> <S4>

    We split the attributes in those that appear at training, and other at test
    the split is strict -- no intersection between train and test

    If in one case the question at training sees 2 attributes for group 2 and 1 for group 1,
    we use the fewer_at_train variable to make sure that for the next template we reverse this.

    :param template: question template, a dictionary
    :param fewer_at_train: bool, to alternate the tuples appearing at tr and ts
    :param extra_bias: bool, if True we impose stricter training bias (relational task only)
    :param same_subtask_input: bool, if True the question equal_size has in input
    <color> equal_size <color> and so on, attributes from the same group
    :return not fewer_at_train: for the next template
    :return training_template:
    :return testing_template:
    """

    original_constraints = template['constraints']  # original constraints
    group_by = grouping_attributes(template)
    # e.g., {1: ['<Z>', '<C>', '<M>', '<S>'], 2: ['<Z2>', '<C2>', '<M2>', '<S2>']}

    constraints_tr = []
    constraints_ts = []

    if same_subtask_input:
        print("Same type of sub-task in input for relational task")
        prev_extracted = 0  # train_split, any attribute is okay
        for id_split in range(2):  # train and test
            keep_extracting = True
            while keep_extracting:
                choice = np.random.choice(group_by[1])  # we extract one attribute for the first group e.g.,color <C>
                if choice != prev_extracted:  # always true at training
                    good_choice = np.zeros(len(group_by), dtype=bool)
                    for id_group, (k_, a_) in enumerate(group_by.items()):  # we verify if <C appears for all groups
                        is_in = [choice[:2] in c_ for c_ in a_]
                        good_choice[id_group] = np.sum(is_in)

                    if np.sum(good_choice) == len(group_by):  # for all groups, <C*> appears
                        keep_extracting = False
                        prev_extracted = choice  # we do not want this attribute at test
                        if id_split == 0:  # save the attribute
                            attr_train = choice
                        else:
                            attr_test = choice

        for k_, a_ in group_by.items():
            # generation of the training and testing splits
            attr_name_train = a_[np.where([attr_train[:2] in c_ for c_ in a_])[0][0]]
            attr_name_test = a_[np.where([attr_test[:2] in c_ for c_ in a_])[0][0]]
            rm_attr_train = a_.copy()
            rm_attr_train.remove(attr_name_train)  # we keep all, but attr_name_train (e.g., C)
            rm_attr_test = a_.copy()
            rm_attr_test.remove(attr_name_test)

            split1 = rm_attr_train  # all the attributes will be forced to be null, with exception of C
            split2 = rm_attr_test

            constraints_tr.append(split1)
            constraints_ts.append(split2)

    else:
        if not extra_bias:
            if fewer_at_train:  # this is done to reduce the amount of excluded attributes
                tmp_ = 0
            else:
                tmp_ = 1

        for k_, a_ in group_by.items():
            # we allow only one attribute to appear in that position
            # this attribute is different from the one fixed for test
            if extra_bias:
                attr_train = np.random.choice(a_, size=1, replace=False)
                rm_attr_train = a_.copy()
                rm_attr_train.remove(attr_train)  # constraints (all but attr_train)

                attr_test = np.random.choice(rm_attr_train, size=1, replace=False)
                # attribute test is selected among all but attr_train
                rm_attr_test = a_.copy()
                rm_attr_test.remove(attr_test)  # constraints (all but attr_test)
                split1 = rm_attr_train
                split2 = rm_attr_test

                constraints_tr.append(split1)
                constraints_ts.append(split2)
            else:
                # in this case, we want to keep the attributes without intersection
                # to increase the amount of examples, we alternate the possible constraints for each questions
                # if template_1 has odd number of parameters (e.g. 3),
                #   we generate a training template_1 which use only one parameter and testing template_1 which uses two
                #   for the next template with odd number of parameters we do the opposite
                split1 = np.random.choice(a_, size=len(a_) // 2, replace=False)  # 1st split of attributes group k_
                split2 = a_.copy()
                [split2.remove(a__) for a__ in split1]  # 2nd split of attributes group k_

                if len(a_) % 2 == 1:  # odd one for training, two for test and vice versa
                    print('fewer at training for first group', fewer_at_train)

                    if tmp_ % 2:  # if [M] in split1, [C,Z] in split2, for next group [M2, Z2] [C2]
                        constraints_tr.append(split1)
                        constraints_ts.append(split2)
                    else:
                        constraints_tr.append(split2)
                        constraints_ts.append(split1)
                    tmp_ += 1
                else:
                    constraints_tr.append(split1)
                    constraints_ts.append(split2)

    # now we need to exclude attributes
    new_templates = []
    # assign the new constraints to the original template
    for constr_splits in [flatten(constraints_tr), flatten(constraints_ts)]:
        template_split = template.copy()
        constr = []
        for c_ in constr_splits:
            constr.append({'params': [c_], 'type': 'NULL'})
        template_split['constraints'] = constr + original_constraints
        new_templates.append(template_split)

    # with not fewer_at_train we alternate -- if we had tr:[S], ts:[M,C], next template will have [Z, S], [M]
    return (not fewer_at_train), new_templates[0], new_templates[1]


def impose_bias_from_relational_question(template):
    original_constraints = template['constraints']  # original constraints
    group_by = grouping_attributes(template)
    print(group_by)
    constraints_tr = []
    constraints_ts = []
    for k_, a_ in group_by.items():
        split1 = np.random.choice(a_, size=1, replace=False)[0] # 1st split of attributes group k_
        tmp_ = a_.copy()
        tmp_.remove(split1)
        split1 = [split1]
        split2 = [np.random.choice(tmp_, size=1, replace=False)[0]]
    return NotImplementedError("Work in progress")


def avoid_stack(template):
    group_by = grouping_attributes(template)
    original_constraints = template['constraints']
    list_attributes = [val_ for val_ in group_by.values()]
    # next we want to use the itertools.product to compute all possible combinations
    # we identify each attribute in the list with a unique id (it becomes the index for the attribute)
    list_attributes_id = []
    count = 0
    for attrs in list_attributes:
        attrs_id = []
        for a_ in attrs:
            attrs_id.append(count)
            count += 1
        list_attributes_id.append(attrs_id)  # now we have all ids, same structure as the list_attributes var
    list_id_tuple = list(itertools.product(*list_attributes_id))
    flat_list_attributes = np.array(flatten(list_attributes))

    list_templates = []
    for id_tuple in list_id_tuple:  # for each combination
        tmp_template = template.copy()
        bm = np.ones(len(flat_list_attributes), dtype=bool)
        bm[list(id_tuple)] = False
        discard_attributes = flat_list_attributes[bm]  # we generate the list of attributes to be set to NULL
        constr = []
        for discard in discard_attributes:
            constr.append({'params': [discard], 'type': 'NULL'})
        tmp_template['constraints'] = constr + original_constraints  # new constraint
        list_templates.append(tmp_template)

    return list_templates


def main(args):
    if len(args.keep_templates) == 0:
        raise ValueError('You need to pass at least one template')

    # if args.tree_structure:
    #     raise NotImplementedError('Work in progress')
    if args.question_bias:
        splits_list = ['train', 'test']
    else:
        splits_list = ['single_split']
    # n_folders_in_templates = len(os.listdir(args.folder_output_templates))
    template_idxs = [int(t_.split('_')[-1]) for t_ in os.listdir(args.folder_output_templates) if 'template' in t_]
    if len(template_idxs) == 0:
        new_template_id = 0
    else:
        new_template_id = np.max(template_idxs)+1

    folder_new_templates = join(args.folder_output_templates, 'template_%i' % new_template_id)
    os.makedirs(folder_new_templates, exist_ok=False)  # do not overwrite templates
    for split in splits_list:
        os.makedirs(join(folder_new_templates, split))

    # json with hyper-parameters for the template generation
    with open(join(folder_new_templates, 'hyper_template_generation.json'), 'w') as f:
        json.dump(args.__dict__, f)

    fewer_train = True
    for question_templates_file in args.keep_templates:  # for each file
        dct_new_templates = {}
        for split in splits_list:
            dct_new_templates[split] = []  # we build a dictionary
            # we save each dictionary in a new file, named as the original, which goes in train or test folder

        question_templates = json.load(open(join(args.folder_CLEVR_templates, question_templates_file), 'rb'))
        for i_ in range(len(question_templates)):
            params = [p_['name'] for p_ in question_templates[i_]['params']]
            include_template = True
            for exclude_arg in args.exclude_questions_with:
                include_template *= not (exclude_arg in params)
                # keep template if the arguments are not in the list of parameters
                # which we want to exclude

            if include_template:
                new_question_templates = {}
                if not args.question_bias:
                    for split_ in splits_list:
                        new_question_templates[split_] = question_templates[i_]
                else:
                    if args.bias_strength.endswith('strict'):
                        if args.bias_strength == 'ultrastrict':
                            extra_bias = True
                        else:
                            extra_bias = False
                            if args.same_subtask_type_in_input:
                                raise ValueError('You are imposing the ultrastrict bias, ' +
                                                 'be sure of that before proceeding further!')
                        print(question_templates[i_]['text'][0])
                        fewer_train, template_tr, template_ts = impose_strict_bias_attributes(question_templates[i_],
                                                                                              fewer_train,
                                                                                              extra_bias=extra_bias,
                                                                                              same_subtask_input=args.same_subtask_type_in_input)
                        out_templates = [template_tr, template_ts] if args.question_bias else [template_tr]
                        print(out_templates)
                    else:
                        raise ValueError('Option not implemented')
                    for split_, template_split in zip(splits_list, out_templates):
                        print('\n', split_, template_split)
                        new_question_templates[split_] = template_split

                if args.no_stack:
                    # if so, set a control
                    for split_ in splits_list:
                        list_templates = avoid_stack(new_question_templates[split_])
                        for t_ in list_templates:
                            dct_new_templates[split_].append(t_)

                else:
                    for split_ in splits_list:
                        dct_new_templates[split_].append(new_question_templates[split_])

        for split in splits_list:
            with open(join(folder_new_templates, split, question_templates_file), 'w') as f:
                json.dump(dct_new_templates[split], f)



