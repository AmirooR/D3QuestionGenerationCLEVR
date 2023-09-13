# Main program for data generation

import argparse
import os
import sys
import importlib
import shutil
from os.path import join
import json
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

parser = argparse.ArgumentParser()
# required parameters
# parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--run', type=str, required=True)

parser.add_argument('--path_dataset', type=str, required=False, default=None,
                    help='When running gen_features and gen_questions' +
                         'path to the dataset is required' +
                         'e.g., as in launch_train_image_generation w/o sub-folder images'
                    )

# args for merge_scenes_files
parser.add_argument('--input_dir', default=None, type=str)
parser.add_argument('--output_file', default=None, type=str)
parser.add_argument('--version', default='1.0')
parser.add_argument('--date', default=(datetime.datetime.now() + relativedelta(years=0)).strftime("%d/%m/%Y"))
parser.add_argument('--license', default='Creative Commons Attribution (CC-BY 4.0')

# args for generate_features_from_imgs
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=64, type=int)

# args for generate_templates
parser.add_argument('--folder_CLEVR_templates',
                    default='CLEVR_dataset_generation/question_generation/CLEVR_1.0_templates',
                    help='Original CLEVR templates')
parser.add_argument('--keep_templates', nargs='+', default=[])
parser.add_argument('--exclude_questions_with', nargs='+', default=[],
                    help='Example: R R2 R3 to exclude relations')
parser.add_argument('--question_bias', action='store_true',
                    help='If False, no bias at the question level')
parser.add_argument('--bias_strength', type=str, choices=['ultrastrict', 'strict', None],
                    default=None,
                    help='ultrastrict: relational q with only one type of input' +
                         'strict: no intersection between train and test questions' +
                         'None: no bias')
parser.add_argument('--same_subtask_type_in_input', action='store_true',
                    help='Use only if question_bias is True')
parser.add_argument('--no_stack', action='store_true',
                    help='If no_stack option is True, we do not allow the concatenation of filters')
parser.add_argument('--folder_output_templates',
                    default='CLEVR_dataset_generation/question_generation/folder_templates',
                    help='Where to save new templates')
parser.add_argument('--manual_comparison', action='store_true')
parser.add_argument('--source_template', type=str, default=None, required=False)

# args for gen_questions
parser.add_argument('--input_vocab_json', type=str, required=False,
                    default='./clevr-iep/vocab.json')
parser.add_argument('--template_name', default='template_0', type=str)
parser.add_argument('--template_dir', default=None, type=str)
parser.add_argument('--metadata_file',
                    default='CLEVR_dataset_generation/question_generation/metadata.json',
                    help="JSON file containing metadata about functions")
parser.add_argument('--synonyms_json',
                    default='CLEVR_dataset_generation/question_generation/synonyms.json',
                    help="JSON file defining synonyms for parameter values")
parser.add_argument('--scene_start_idx', default=0, type=int)
parser.add_argument('--num_scenes', default=0, type=int)
parser.add_argument('--templates_per_image', default=10, type=int)
parser.add_argument('--instances_per_template', default=1, type=int)
parser.add_argument('--reset_counts_every', default=250, type=int,
                    help="How often to reset template and answer counts." +
                         "Higher values will result in flatter distributions.")
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--time_dfs', action='store_true')
parser.add_argument('--profile', action='store_true')
# args for clevr-iep.scripts.preprocess_questions
parser.add_argument('--mode', default='prefix', choices=['chain', 'prefix', 'postfix'])
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)
parser.add_argument('--output_vocab_json', default='')

parser.add_argument('--allow_no_stack', action='store_true')
parser.add_argument('--force_and', action='store_true')
parser.add_argument('--define_stack', type=int, default=1)

parser.add_argument('--extract_data_from', default=None, type=str)
parser.add_argument('--output_data_path', default=None, type=str)
parser.add_argument('--size_dataset', default=None, type=int)
parser.add_argument('--same_size_as', default=None, type=str)

parser.add_argument('--path_input_questions_json', type=str, required=False)
parser.add_argument('--path_input_scenes', type=str, required=False)
parser.add_argument('--path_input_questions_h5', type=str, required=False)
parser.add_argument('--output_questions_folder', type=str, required=False)
parser.add_argument('--curriculum', type=str, required=False, choices=['pg', 'all'])


# create subset
parser.add_argument('--name_h5_file', type=str, required=False)
parser.add_argument('--n_examples', type=int, required=False)

# merge phase 1 + phase 2
parser.add_argument('--path_dataset_phase1', type=str, required=False)
parser.add_argument('--path_dataset_phase2', type=str, required=False)
parser.add_argument('--output_path', type=str, required=False)

FLAGS = parser.parse_args()


def merge_scenes_files(id=0):
    if FLAGS.input_dir is None and FLAGS.output_file is None:
        raise ValueError("You need to pass the file scenes containing all scenes or the folder to merge scenes")
    from CLEVR_dataset_generation.image_generation import collect_scenes
    print('Ready to merge')
    collect_scenes.main(FLAGS)
    print('Done!')


def generate_features_from_imgs(id=0):
    """
    Given the image folder and the scene folder for a CLEVR type of dataset,
    we merge the scenes files in a unique file
    and we extract the features using the pretrained ResNet101.
    """
    sys.path.append('clevr-iep')
    from scripts import extract_features
    if FLAGS.path_dataset is None:
        raise ValueError("Path to the dataset folder is missing")
    path_images_folders = FLAGS.path_dataset
    images_folders = [imgf_ for imgf_ in os.listdir(path_images_folders) if
                      imgf_.endswith('images') and os.path.isdir(join(path_images_folders, imgf_))]
    for imgf_ in images_folders:
        split_name = imgf_.split('images')[0]  # name of the split, e.g., train_A_
        FLAGS.input_image_dir = join(path_images_folders, imgf_)
        FLAGS.output_h5_file = join(path_images_folders, '%sfeatures.h5' % split_name)
        extract_features.main(FLAGS)


def generate_templates():
    """
    Template generation based on the CLEVR templates.
    Question bias can be imposed at the attribute level, not at the relation level.

    If aiming at having bias at the relation level, new code is needed to constraint the
    templates before running generate_templates_constrain_attributes.main()
    """
    os.makedirs(FLAGS.folder_output_templates, exist_ok=True)

    if FLAGS.manual_comparison:
        if len(os.listdir(FLAGS.folder_output_templates)) == 0:
            id_new_templates = 0
        else:
            id_new_templates = 1 + np.max([int(t_.split('_')[-1])
                                           for t_ in os.listdir(FLAGS.folder_output_templates)])
        folder_template = join(FLAGS.folder_output_templates, 'template_%i' % id_new_templates)
        os.makedirs(folder_template)
        from CLEVR_dataset_generation.question_generation import generate_templates_comparison
        generate_templates_comparison.run(folder_template)
        return

    from CLEVR_dataset_generation.question_generation import generate_templates_constrain_attributes
    if FLAGS.question_bias and FLAGS.bias_strength is None:
        raise ValueError('You need to specify the entity of bias')

    if len(FLAGS.exclude_questions_with) > 0:
        parsed_list = ['<' + q_ + '>' for q_ in FLAGS.exclude_questions_with]
        FLAGS.exclude_questions_with = parsed_list

    # pass all parameters
    params_list = []
    for file in os.listdir(FLAGS.folder_CLEVR_templates):
        templates_f = json.load(open(join(FLAGS.folder_CLEVR_templates, file), 'rb'))
        for t_ in templates_f:
            for param in t_['params']:
                params_list.append(param['name'])
    FLAGS.params_list = list(np.unique(np.array(params_list)))
    generate_templates_constrain_attributes.main(FLAGS)


def generate_questions_files(id=0):
    """ Question generation relying on templates """
    from CLEVR_dataset_generation.question_generation import generate_questions, convert_type_to_function
    if FLAGS.path_dataset is None:
        raise ValueError("Path to the dataset folder is missing")

    if FLAGS.template_dir is None:
        FLAGS.template_dir = join(FLAGS.folder_output_templates, FLAGS.template_name)

    path_scenes_files = join(FLAGS.path_dataset)
    path_question_folder = join(FLAGS.path_dataset, 'questions')
    os.makedirs(path_question_folder, exist_ok=True)
    path_question_folder_templates = join(path_question_folder, FLAGS.template_name)
    os.makedirs(path_question_folder_templates, exist_ok=True)

    scenes_files = [f_ for f_ in os.listdir(path_scenes_files) if f_.endswith('scenes.json')]
    print(scenes_files)
    # there are sub-folders for templates to be used in-distribution and ood
    splits = [dir_ for dir_ in os.listdir(FLAGS.template_dir)
              if os.path.isdir(join(FLAGS.template_dir, dir_))]
    if len(splits) == 1:
        split_tr_ts = False
    elif 'train' in os.listdir(FLAGS.template_dir) and 'test' in os.listdir(FLAGS.template_dir):
        split_tr_ts = True
    else:
        raise ValueError("Option not considered")

    sup_folder = FLAGS.template_dir

    json_question_files = []
    for sf_ in scenes_files:  # for all scenes files
        # FLAGS.input_scene_file = join(path_scenes_files, sf_)
        FLAGS.input_scene_file = join(FLAGS.path_dataset, sf_)

        if split_tr_ts:  # we have two different families of templates (in-d and ood)
            splits_allowed = ['train'] if sf_.split('scenes')[0].startswith('train') else ['train', 'test']
            for split in splits_allowed:
                bias_q = 'ind' if split == 'train' else 'ood'
                if FLAGS.num_scenes > 0:
                    folder_qst_from_scenes = '%ssplit_qst_gen' % sf_.split('scenes')[0]
                    os.makedirs(join(path_question_folder_templates, folder_qst_from_scenes), exist_ok=True)
                    name_output_q = '%s/%s%s_%i_questions.json' % (folder_qst_from_scenes,
                                                                   sf_.split('scenes')[0],
                                                                   bias_q,
                                                                   FLAGS.scene_start_idx)
                else:
                    name_output_q = '%s%s_questions.json' % (sf_.split('scenes')[0], bias_q)
                FLAGS.output_questions_file = join(path_question_folder_templates, name_output_q)
                FLAGS.template_dir = join(sup_folder, split)
                json_question_files.append(FLAGS.output_questions_file)
                generate_questions.main(FLAGS)
                convert_type_to_function.convert(FLAGS.output_questions_file)
        else:
            FLAGS.template_dir = join(sup_folder, splits[0])
            if FLAGS.num_scenes > 0:
                folder_qst_from_scenes = '%ssplit_qst_gen' % sf_.split('scenes')[0]
                os.makedirs(join(path_question_folder_templates, folder_qst_from_scenes), exist_ok=True)
                name_output_q = '%s/%s%i_questions.json' % (folder_qst_from_scenes,
                    sf_.split('scenes')[0], FLAGS.scene_start_idx)
            else:
                name_output_q = '%squestions.json' % sf_.split('scenes')[0]
            FLAGS.output_questions_file = join(path_question_folder_templates, name_output_q)
            json_question_files.append(FLAGS.output_questions_file)
            generate_questions.main(FLAGS)
            convert_type_to_function.convert(FLAGS.output_questions_file)

    shutil.copyfile(FLAGS.input_vocab_json, join(path_question_folder_templates, 'vocab.json'))
    if FLAGS.num_scenes == 0:
        _preprocess_questions(id, json_question_files)


def merge_and_preprocess_questions(id=0):
    """ To be used in case FLAGS.num_scenes != 0"""
    from CLEVR_dataset_generation.question_generation import merge_questions
    for folder_ in os.listdir(FLAGS.path_dataset):
        if folder_.endswith('split_qst_gen'):
            path_file = merge_questions.merging_question_files(join(FLAGS.path_dataset, folder_),
                                                       output_path=FLAGS.path_dataset,
                                                       num_scenes=FLAGS.num_scenes)
            _preprocess_questions(id, [path_file])


def _preprocess_questions(id, json_question_files):
    sys.path.append('clevr-iep')
    from scripts import preprocess_questions
    for input_q_file in json_question_files:
        FLAGS.input_questions_json = input_q_file
        filename = input_q_file.split('/')[-1]
        FLAGS.output_h5_file = join(os.path.dirname(input_q_file), '%s.h5' % filename.split('.')[0])
        preprocess_questions.main(FLAGS)


def change_program_elemental_questions(id=0):
    """ Force stack for concatenated filters only """
    from CLEVR_dataset_generation.question_generation import generate_elemental_questions_with_stack
    questions_path = join(FLAGS.path_dataset, 'questions', FLAGS.template_name)
    if FLAGS.allow_no_stack and not FLAGS.force_and:
        print('Questions as originals')
        return
    generate_elemental_questions_with_stack.force_program_structure(questions_path,
                                                                    FLAGS.input_vocab_json,
                                                                    allow_no_stack=FLAGS.allow_no_stack,
                                                                    force_and=FLAGS.force_and)


def avoid_stack_given_templates(id=0):
    source_template_name = FLAGS.source_template
    id_new_templates = 1 + np.max([int(t_.split('_')[-1])
                                   for t_ in os.listdir(FLAGS.folder_output_templates) if 'template' in t_])
    folder_template = join(FLAGS.folder_output_templates, 'template_%i' % id_new_templates)
    os.makedirs(folder_template)
    with open(join(folder_template, 'hyper_template_generation.json'), 'w') as f:
        json.dump(FLAGS.__dict__, f)
    from CLEVR_dataset_generation.question_generation import avoid_stack_in_template
    avoid_stack_in_template.avoid_stack(join(FLAGS.folder_output_templates, source_template_name),
                                        folder_template)


def split_dataset_for_curriculum(id=0):
    from scripts.divide_curriculum import divide_curriculum
    divide_curriculum(path_input_questions_json=FLAGS.path_input_questions_json,
                      path_input_scenes=FLAGS.path_input_scenes,
                      path_input_questions_h5=FLAGS.path_input_questions_h5,
                      output_questions_folder=FLAGS.output_questions_folder,
                      curriculum=FLAGS.curriculum)


def create_subset(id=0):
    import scripts.create_clevr_data_from_other as reduce_dataset
    path_to_input_h5_questions = join(FLAGS.path_dataset, 'questions', FLAGS.template_name)
    path_to_output_h5_questions = join(FLAGS.path_dataset, 'questions',
                                       '%s/subset_%i' % (FLAGS.template_name, FLAGS.n_examples))
    os.makedirs(path_to_output_h5_questions)
    reduce_dataset.main(path_input_file=join(path_to_input_h5_questions, FLAGS.name_h5_file),
                        path_output_file=join(path_to_output_h5_questions, FLAGS.name_h5_file),
                        n_examples=FLAGS.n_examples)
    shutil.copyfile(FLAGS.input_vocab_json,
                    join(path_to_output_h5_questions, 'vocab.json'))


def merge_datasets(id=0):
    from scripts.merge_phase1_phase2 import merge
    merge(FLAGS.path_dataset_phase1, FLAGS.path_dataset_phase2,
          FLAGS.output_path, FLAGS.input_vocab_json,
          FLAGS.name_h5_file)


def remove_unique_comparison(id=0):
    from scripts.remove_unique_in_comparison import remove_unique
    remove_unique(FLAGS.path_input_dataset_w_unique, FLAGS.path_output_dataset_wo_unique)


switcher = {
    'gen_templates': generate_templates,
    'merge_scenes': merge_scenes_files,
    'merge_questions': merge_and_preprocess_questions,
    'gen_features': generate_features_from_imgs,
    'gen_questions': generate_questions_files,
    'change_elem_questions': change_program_elemental_questions,
    'avoid_stack_in_template': avoid_stack_given_templates,
    'create_curriculum': split_dataset_for_curriculum,
    'create_subset': create_subset,
    'merge_datasets': merge_datasets,
    'remove_unique': remove_unique_comparison
}

switcher[FLAGS.run]()