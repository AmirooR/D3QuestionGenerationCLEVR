dataset_root="data/two_hop_datasets/questions"

augment_and_convert_to_h5()
{
  unbiased_template_name=$1
  unbiased_question_file=$2
  biased_template_name=$3
  biased_question_file=$4
  output_template_name=$5
  output_question_file=$6
  fraction=$7

  unbiased_questions="${dataset_root}/${unbiased_template_name}/${unbiased_question_file}"
  biased_questions="${dataset_root}/${biased_template_name}/${biased_question_file}"
  output_questions="${dataset_root}/${output_template_name}/${output_question_file}"
  vocab_file="${dataset_root}/${biased_template_name}/vocab.json"

  mkdir -p ${dataset_root}/${output_template_name}

  echo "AUGMENTING Questions of ${unbiased_questions} and ${biased_questions}"
  python augment_questions.py \
    --unbiased_questions "${unbiased_questions}.json" \
    --biased_questions "${biased_questions}.json" \
    --output_questions "${output_questions}.json" \
    --fraction ${fraction}

  echo "CONVERTING ${output_questions} TO h5"

  cd clevr-iep && python scripts/preprocess_questions.py --input_questions_json "${output_questions}.json" --output_h5_file "${output_questions}.h5" --input_vocab_json "${vocab_file}" && cd ..
}


# train ind one (30% of 1HopFull with 2HopA)
unbiased_template_name="1HopFull"
unbiased_question_file="CLEVR_trainA_questions"
biased_template_name="2HopA"
biased_question_file="CLEVR_trainA_ind_questions"
output_template_name="D3_2HopA/1HopFull"
output_question_file="train_questions"
fraction="0.3"

augment_and_convert_to_h5 $unbiased_template_name $unbiased_question_file $biased_template_name $biased_question_file $output_template_name $output_question_file $fraction

# val ind one
unbiased_question_file="CLEVR_valA_questions"
biased_question_file="CLEVR_valA_ind_questions"
output_question_file="val_questions"

augment_and_convert_to_h5 $unbiased_template_name $unbiased_question_file $biased_template_name $biased_question_file $output_template_name $output_question_file $fraction


