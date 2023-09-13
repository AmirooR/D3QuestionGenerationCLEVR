import json
import os
from os.path import join


def param_is_attribute(param_name):
    if param_name.startswith('<Z'):
        return True
    elif param_name.startswith('<M'):
        return True
    elif param_name.startswith('<C'):
        return True
    elif param_name.startswith('<S'):
        return True
    else:
        return False


def add_attributes_to_avoid_stack(constraints_list):
    new_constraints_list = []
    free_params = ['<M>', '<S>', '<C>', '<Z>']
    for c_ in constraints_list:
        free_params.remove(c_)
    if len(free_params) == 2:
        for fp in free_params:
            new_constraint = constraints_list.copy()
            new_constraint.append(fp)
            new_constraints_list.append(new_constraint)
    else:
        raise NotImplementedError("The case with one constrained attributes over four" +
                                  " has not been considered yet")
    return new_constraints_list


def avoid_stack(source_template, destination_template):
    """
    Function to adapt templates in the case of one group of object with question bias.
    And force to have a single attribute instead of stack of attributes. """
    splits_list = [s_ for s_ in os.listdir(source_template) if os.path.isdir(join(source_template, s_))]
    for split in splits_list:
        os.makedirs(join(destination_template, split))
        for f in os.listdir(join(source_template, split)):
            templates = json.load(open(join(source_template, split, f), 'rb'))
            list_new_templates = []
            for t in templates:
                constraints_list = [c_['params'][0] for c_ in t['constraints']
                                    if c_['type'] == 'NULL']
                params_list = [p_ for p_ in t['params']
                               if param_is_attribute(p_['name'])]
                if len(params_list) > 4:
                    raise NotImplementedError('This functionality is not implemented for more than' +
                                              ' one group of objects.')
                templates_no_stack = t.copy()
                if len(constraints_list) < 3:
                    print(t['text'][0])
                    new_constraint_list = add_attributes_to_avoid_stack(constraints_list)
                    print(new_constraint_list)
                    for tuple_constraints in new_constraint_list:
                        print(tuple_constraints)
                        templates_no_stack = t.copy()
                        templates_no_stack['constraints'] = []
                        for c_att in tuple_constraints:
                            templates_no_stack['constraints'].append({'params': [c_att], 'type': 'NULL'})
                        print(templates_no_stack['constraints'])
                        list_new_templates.append(templates_no_stack)

                else:
                    list_new_templates.append(templates_no_stack)
            print(list_new_templates)
            json.dump(list_new_templates, open(join(destination_template, split, f), 'w'))
