# Generate Customized CLEVR Datasets
This code is a fork from
https://github.com/facebookresearch/clevr-dataset-gen (CLEVR dataset generation) and 
https://github.com/facebookresearch/clevr-iep (features generation and base models). 

### Features generation
Once you have your images, you can extract features using a pretrained ResNet-101.
To do so, run the commands in `python3 main.py --run gen_features`. 
Specify the path using `--path_dataset` which contains the folders named `*_*_images` like `train_A_images`.

### Templates generation
You can generate the template by running the commands in `python3 main.py --run gen_templates`.
We have provided the 2Hop templates in `CLEVR_dataset_generation/question_generation/two_hop_templates`. For these templates, we have added the constraints in their respective json files. The constraints are defined in `CLEVR_dataset_generation/question_generation/generate_questions.py` after line 270.

### Questions generation
To generate questions, look at the commands in `launch_twohop_question_generation.sh` as a sample. 

There might be a case (less than 1% of questions) in 3/2Hop OOD that might have some overlap with 2Hop A. We used the `verify_two_hop.py` to generate a mask to exclude these questions from the OOD test sets when we compute the accuracies.

### D3 sets generation
To generate a sample D3 set, look at the commands in `augment_questions.sh` as a sample.

### Question complexity distribution generation
To generate datasets for question complexity distributions run the following:
First you need to merge the generated json question files. It randomly samples 400k questions from each of the 3 question files and generates a new question file with a total 1.2M questions:

```
python merge_question_files.py --question_json_files data/two_hop_datasets/questions/2HopA/CLEVR_trainA_questions.json  data/two_hop_datasets/questions/0HopA/CLEVR_trainA_questions.json data/two_hop_datasets/questions/1HopFull/CLEVR_trainA_questions.json --output_json_file  data/two_hop_datasets/questions/LARGE_wide/CLEVR_trainA_questions.json --num_samples 400000
```

Then, generate the datasets with different distributions and a total of 800k questions using a catalog file:

```
python gen_dataset_dists.py --wide_json_file data/two_hop_datasets/questions/LARGE_wide/CLEVR_trainA_questions.json --catalog sample_dataset_catalog.json --output data/two_hop_datasets/wide_subsets/800k --total 800000
```

Note that the questions can then be fed to the CLOSURE for running train and tests.

## Cite

If you make use of this code in your own work, please cite our paper:

```
@inproceedings{rahimi2023D3,
  title={D3: Data Diversity Design for Systematic Generalization in Visual Question Answering},
  author={Rahimi, Amir and D'Amario, Vanessa and Yamada, Moyuru and Takemoto, Kentaro and Sasaki, Tomotake and Boix, Xavier},
  booktitle={Arxiv},
  year={2023}
}
```
