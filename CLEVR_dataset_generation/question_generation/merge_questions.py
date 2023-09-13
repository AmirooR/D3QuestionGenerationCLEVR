import json
import os
from os.path import join


def merging_question_files(input_path_multiple_questions,
                           output_path,
                           num_scenes=1000):
    """ Merge the json files in input_path_multiple_questions
     into a unique json and save the file.
    :param input_path_multiple_questions: str, path to the single scenes files
    :param output_path: path to the json file containing all scenes
    """
    dct = {}
    dct['questions'] = []
    filename_list = os.listdir(input_path_multiple_questions)
    start, end = filename_list[0].split('%i_' % 0)[0], filename_list[0].split('%i_' % 0)[-1]

    for id_, _ in enumerate(filename_list):
        id_file = id_ * num_scenes
        file_ = start + '%i_' % id_file + end
        json_questions = json.load(open(join(input_path_multiple_questions, file_), 'rb'))
        if id_ == 0:
            dct['info'] = json_questions['info']

        dct['questions'].extend(json_questions['questions'])

    for i, q in enumerate(dct['questions']):
        q['question_index'] = i

    filename = start + end
    with open(join(output_path, filename), 'w') as f:
        json.dump(dct, f)

    return join(output_path, filename)
