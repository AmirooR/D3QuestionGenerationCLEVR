#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER


generate_questions()
{
  name=$1
  template_name=$2
  python3 main.py \
  --run gen_questions \
  --path_dataset "data/${name}_datasets" \
  --folder_output_templates "CLEVR_dataset_generation/question_generation/${name}_templates" \
  --template_name $template_name
}

generate_questions_large()
{
  name=$1
  template_name=$2
  python3 main.py \
  --run gen_questions \
  --path_dataset "data/${name}_datasets" \
  --folder_output_templates "CLEVR_dataset_generation/question_generation/${name}_templates" \
  --template_name $template_name --templates_per_image 20 --instances_per_template 5
}

name="two_hop"
generate_questions_large $name "1HopFull"

