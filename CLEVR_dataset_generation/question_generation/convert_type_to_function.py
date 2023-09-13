import json
from os.path import join, dirname


def convert(path_json_file):
    questions = json.load(open(path_json_file, 'rb'))
    for kkk in range(len(questions['questions'])):
        for node in questions['questions'][kkk]['program']:
            node['function'] = node['type']
            del node['type']

    with open(path_json_file, 'w') as f:
        json.dump(questions, f)


def main():
    root_path = 'PATH_TO_DATASET_ROOT'
    folder_data = 'objs_2_5_compositional_with_BQ/questions'

    for filename in ['CoGenT_test_A_BQ_bias_q.json',
                     'CoGenT_test_A_bias_q.json',
                     'CoGenT_test_B_NBQ_bias_q.json',
                     'CoGenT_test_B_bias_q.json',
                     'CoGenT_train_A_bias_q.json',  # ,
                     'CoGenT_val_A_bias_q.json',
                     ]:

        questions = json.load(open(join(root_path, folder_data, filename), 'rb'))
        for kkk in range(len(questions['questions'])):
            for node in questions['questions'][kkk]['program']:
                node['function'] = node['type']
                del node['type']

        with open(join(root_path, folder_data, 'functions_' + filename), 'w') as f:
            json.dump(questions, f)


if __name__ == "__main__":
    main()
