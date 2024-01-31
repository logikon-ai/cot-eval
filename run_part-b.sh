#!bin/bash

set -a # automatically export all variables
source config.env
source ../.env
set +a


set -e # exit on error

if [[ -z "${HUGGINGFACEHUB_API_TOKEN}" ]]; then
  echo "HUGGINGFACEHUB_API_TOKEN not found. Please set it in .env file."
  exit 1
fi

huggingface-cli login --token $HUGGINGFACEHUB_API_TOKEN


# lookup model to-be evaluated in files generated in subscript-a

model=$(cat next_model.json | jq -r .model)
revision=$(cat next_model.json | jq -r .revision)
precision=$(cat next_model.json | jq -r .precision)
echo "Model to evaluate: $model : $revision. Precision: $precision"

configkeys=$(cat config_keys.txt)  # format is "config1,config2,config3"
echo "Found configs: $configkeys"


# create lm-eval-harness tasks
# a 'harness task' defines how to evaluate a given model on a given task,
# specifically whether to include the model's reasoning traces or not
python scripts/create_lm_eval_harness_tasks.py \
    --configs $configkeys \
    --output_dir eleuther/tasks/logikon \
    --keys_file ./lm_eval_harness_tasks.json
harness_tasks_base=$(cat lm_eval_harness_tasks.json | jq -r .base) # format is "task1,task2,task3"
harness_tasks_cot=$(cat lm_eval_harness_tasks.json | jq -r .cot) # format is "task1,task2,task3"
echo "Created lm-eval-harness tasks base: $harness_tasks_base" # no cot
echo "Created lm-eval-harness tasks cot: $harness_tasks_cot"


timestamp=$(date +"%y-%m-%d-%T")

# run lm-eval originial BASE (unperturbed) for each task
if [ "$DO_BASEEVAL" = true ] ; then
    arrTASKS=(${TASKS//,/ })
    basetasks=$(printf "%s_base," "${arrTASKS[@]}")
    basetasks=${basetasks:0:-1}
    output_path=$OUTPUT_DIR/${model}/orig/results_${timestamp}.json
    if [ -f $output_path ]; then
        echo "Outputfile $FILE exists. Skipping eval of $basetasks."
    else
        lm-eval --model vllm \
            --model_args pretrained=${model},revision=${revision},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE,max_length=$MAX_LENGTH \
            --tasks $basetasks \
            --num_fewshot 0 \
            --batch_size auto \
            --output_path $output_path \
            --include_path ./eleuther/tasks/logikon
    fi
fi


## run lm evaluation harness for each of the tasks
# without reasoning traces
lm-eval --model vllm \
    --model_args pretrained=${model},revision=${revision},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE,max_length=$MAX_LENGTH \
    --tasks ${harness_tasks_base} \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path $OUTPUT_DIR/${model}/base/${timestamp}.json \
    --include_path ./eleuther/tasks/logikon
# with reasoning traces
lm-eval --model vllm \
    --model_args pretrained=${model},revision=${revision},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE,max_length=$MAX_LENGTH \
    --tasks ${harness_tasks_cot} \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path $OUTPUT_DIR/${model}/cot/${timestamp}.json \
    --include_path ./eleuther/tasks/logikon


# collect and upload results
python scripts/upload_results.py \
    --model $model \
    --revision $revision \
    --precision $precision \
    --tasks $TASKS \
    --timestamp $timestamp \
    --output_dir $OUTPUT_DIR
#    --harness_tasks_base $harness_tasks_base \
#    --harness_tasks_cot $harness_tasks_cot

# cleanup
rm ./next_model.json
rm ./config_keys.txt
rm ./lm_eval_harness_tasks.json
