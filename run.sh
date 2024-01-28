#!bin/bash

CHAINS="HandsOn,ReflectBeforeRun" # "HandsOn"
MODELKWARGS='[{temperature: .3, top_k: 100, top_p: .95},{temperature: 0},{use_beam_search: true, best_of: 1, n: 4}]'  # YAML format
TASKS="logiqa,logiqa2,lsat-ar,lsat-rc,lsat-lr"
OUTPUT_DIR="./eleuther/output"
CONFIGS_DIR="src/cot_eval/configs"
TRUST_REMOTE_CODE=true
MAX_LENGTH=4096
DO_BASEEVAL=true

set -a # automatically export all variables
source ../.env
set +a

huggingface-cli login --token $HUGGINGFACEHUB_API_TOKEN

# lookup model to-be evaluated
python scripts/lookup_pending_model.py --keys_file ./next_model.json
model=$(cat next_model.json | jq -r .model)
revision=$(cat next_model.json | jq -r .revision)
precision=$(cat next_model.json | jq -r .precision)
echo "Model to evaluate: $model : $revision. Precision: $precision"

# create configs
# a 'config' defines how reasoning traces are generated for a given task
python scripts/create_cot_configs.py \
    --model $model \
    --revision $revision \
    --chains $CHAINS \
    --model_kwargs "$MODELKWARGS" \
    --tasks $TASKS \
    --output_dir $CONFIGS_DIR \
    --keys_file ./config_keys.txt
configkeys=$(cat config_keys.txt)  # format is "config1,config2,config3"
echo "Created configs: $configkeys"


# run cot_eval to create reasoning traces for every config (model and task)
# reasoning traces are uploaded to huggingface hub
arr_configkeys=(${configkeys//,/ })
for config in "${arr_configkeys[@]}"
do
    cot-eval \
        --config $CONFIGS_DIR/$config.yaml \
        --hftoken $HUGGINGFACEHUB_API_TOKEN
done


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

# run lm-eval BASE (unperturbed) for each task
if [ "$DO_BASEEVAL" = true ] ; then
    arrTASKS=(${TASKS//,/ })
    for task in "${arrTASKS[@]}"
    do
        output_path=$OUTPUT_DIR/${model}/orig/results_${timestamp}.json
        if [ -f $output_path ]; then
            echo "Outputfile $FILE exists. Skipping task $task."
        else
            lm-eval --model vllm \
                --model_args pretrained=${model},revision=${revision},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE,max_length=$MAX_LENGTH \
                --tasks ${task}_base \
                --num_fewshot 0 \
                --batch_size auto \
                --output_path $output_path \
                --include_path ./eleuther/tasks/logikon
        fi
    done
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

