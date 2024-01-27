#!bin/bash

CHAINS="HandsOn" # "HandsOn,PlanAndExecute"
MODELKWARGS='[{temperature: .3, top_k: 100, top_p: .95},{temperature: 0},{use_beam_search: true, best_of: 1, n: 4}]'  # YAML format
TASKS="logiqa,logiqa2,lsat-ar,lsat-rc,lsat-lr"
OUTPUT_DIR="./eleuther/output"
TRUST_REMOTE_CODE=true
MAX_LENGTH=4096
DO_BASEEVAL=true

# lookup model to-be evaluated
python scripts/lookup_pending_model.py --keys_file ./next_model.json
model=$(cat next_model.json | jq -r .model)
revision=$(cat next_model.json | jq -r .revision)
modelbase="$(basename -- $model)"
echo "Model to evaluate: $model : $revision"

# create configs
python scripts/create_cot_configs.py \
    --model $model \
    --revision $revision \
    --chains $CHAINS \
    --model_kwargs "$MODELKWARGS" \
    --tasks $TASKS \
    --output_dir src/cot_eval/configs \
    --keys_file ./config_keys.txt
configkeys=$(cat config_keys.txt)  # format is "config1,config2,config3"
echo "Created configs: $configkeys"


# run cot_eval to create reasoning traces
arr_configkeys=(${configkeys//,/ })
for config in "${arr_configkeys[@]}"
do
    python cot_eval \
        --config $config \
        --hftoken $HUGGINGFACEHUB_API_TOKEN
done


# create lm-eval-harness tasks
## includes tasks with and without cot traces
python scripts/create_lm_eval_harness_tasks.py \
    --configs $configkeys \
    --output_dir eleuther/tasks/logikon \
    --keys_file ./lm_eval_harness_tasks.txt
harness_tasks=$(cat lm_eval_harness_tasks.txt)  # format is "task1,task2,task3"
echo "Created lm-eval-harness tasks: $harness_tasks"


# run lm-eval BASE for each of the tasks
if [ "$DO_BASEEVAL" = true ] ; then
    arrTASKS=(${TASKS//,/ })
    for task in "${arrTASKS[@]}"
    do
        output_path=$OUTPUT_DIR/${modelbase}/${task}_base.json
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
arr_harness_tasks=(${harness_tasks//,/ })
for task in "${arr_harness_tasks[@]}"
do
    lm-eval --model vllm \
        --model_args pretrained=${model},revision=${revision},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE,max_length=$MAX_LENGTH \
        --tasks ${task} \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path $OUTPUT_DIR/${modelbase}/${task}.json \
        --include_path ./eleuther/tasks/logikon
done


# collect and upload results
python scripts/upload_results.py \
    --model $model \
    --revision $revision \
    --tasks $TASKS \
    --harness_tasks $harness_tasks \
    --output_dir $OUTPUT_DIR \
    --hftoken $HUGGINGFACEHUB_API_TOKEN

# cleanup
rm ./next_model.json
rm ./config_keys.txt
rm ./lm_eval_harness_tasks.txt

