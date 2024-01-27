#!bin/bash

CHAINS="HandsOn" # "HandsOn,PlanAndExecute"
MODELKWARGS="[{temperature: .3, top_k: 100, top_p: .95},{temperature: 0},{use_beam_search: true, best_of: 1, n: 4}]"  # YAML format
TASKS="logiqa,logiqa2,lsat-ar,lsat-rc,lsat-lr"
OUTPUT_DIR="./eleuther/output"
TRUST_REMOTE_CODE=true
MAX_LENGTH=4096
DO_BASEEVAL=true
modelbase="$(basename -- $MODEL)"


# lookup model to-be evaluated
python scripts/lookup_pending_model.py &
MODEL=$!
echo "Model to be evaluated: $MODEL"

#MODEL="mistralai/Mistral-7B-Instruct-v0.2"


# create configs
python scripts/create_config.py \
    --model $MODEL \
    --chain $CHAINS \
    --model_kwargs $MODELKWARGS \
    --tasks $TASKS \
    --output_dir src/cot_eval/configs &
CONFIGNAMES=$!
echo "Created configs: $CONFIGNAMES"


# create lm-eval-harness tasks
## includes tasks with and without cot traces
python scripts/create_lm_eval_harness_tasks.py \
    --configs $CONFIGNAMES \
    --output_dir eleuther/tasks/logikon
HARNESS_TASKS=$!
echo "Created lm-eval-harness tasks: $HARNESS_TASKS"


# run cot_eval to create reasoning traces
arrCONFIGNAMES=(${CONFIGNAMES//,/ })
for config in "${arrCONFIGNAMES[@]}"
do
    python cot_eval \
        --config $config \
        --hftoken $HUGGINGFACEHUB_API_TOKEN
done

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
                --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE,max_length=$MAX_LENGTH \
                --tasks ${task}_base \
                --num_fewshot 0 \
                --batch_size auto \
                --output_path $output_path \
                --include_path ./eleuther/tasks/logikon
        fi
    done
fi


## run lm evaluation harness for each of the tasks
arrHARNESS_TASKS=(${HARNESS_TASKS//,/ })
for task in "${arrHARNESS_TASKS[@]}"
do
    lm-eval --model vllm \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE,max_length=$MAX_LENGTH \
        --tasks ${task} \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path $OUTPUT_DIR/${modelbase}/${task}.json \
        --include_path ./eleuther/tasks/logikon
done


# collect and upload results
python scripts/upload_results.py \
    --model $MODEL \
    --tasks $TASKS \
    --harness_tasks $HARNESS_TASKS \
    --output_dir $OUTPUT_DIR \
    --hftoken $HUGGINGFACEHUB_API_TOKEN