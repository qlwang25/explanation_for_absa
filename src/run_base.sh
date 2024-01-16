#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

domains=('lap14') 
data='../data/'
output='../data/run_out/'

for domain in  ${domains[@]};
do
    echo "####################### ${model_name} ${domain} #######################:"
    python -B run_classification.py \
        --data_dir "${data}${domain}" --output_dir "${output}${domain}"  \
        --train_batch_size 48  --eval_batch_size 30 --learning_rate 3e-5 --max_seq_length 256  \
        --plm_model bert_base --seed 42 --do_train --do_eval 
    printf "\n\n"

done