#!bin/bash

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
CHAIN="HandsOn"
MODELKWARGS="{temperature: .3, top_k: 100, top_p: .95}"  # YAML format
TASKS="logiqa,logiqa2,lsat-ar,lsat-rc,lsat-lr"
OUTPUT_DIR="./eleuther/output"
TRUST_REMOTE_CODE=true
DO_BASEEVAL=true
modelbase="$(basename -- $MODEL)"

# create config
python scripts/create_config.py \
    --model $MODEL \
    --chain $CHAIN \
    --model_kwargs $MODELKWARGS \
    --tasks $tasks \
    --output_dir src/cot_eval/configs &
CONFIGNAME=$!
echo "Created config: $CONFIGNAME"


# create lm-eval-harness tasks
## includes tasks with and without cot traces
python scripts/create_lm_eval_harness_tasks.py \
    --config ./$CONFIGNAME \
    --output_dir eleuther/tasks/logikon
HARNESS_TASKS=$!
echo "Created lm-eval-harness tasks: $HARNESS_TASKS"


# run cot_eval to create reasoning traces
python cot_eval \
    --config $CONFIGNAME \
    --hftoken $HUGGINGFACEHUB_API_TOKEN


# run lm_eval BASE for each of the tasks
if [ "$DO_BASEEVAL" = true ] ; then
    arrTASKS=(${TASKS//,/ })
    for task in "${arrTASKS[@]}"
    do
        output_path=$OUTPUT_DIR/${modelbase}/${task}_base.json
        if [ -f $output_path ]; then
            echo "Outputfile $FILE exists. Skipping task $task."
        else
            lm_eval --model vllm \
                --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE \
                --tasks ${task}_base \
                --num_fewshot 0 \
                --batch_size auto \
                --output_path $OUTPUT_DIR/${modelbase}/${task}_base.json \
                --include_path ./eleuther/tasks/logikon
        fi
    done
fi


## run lm evaluation harness for each of the tasks
arrHARNESS_TASKS=(${HARNESS_TASKS//,/ })
for task in "${arrHARNESS_TASKS[@]}"
do
    lm_eval --model vllm \
        --model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=$TRUST_REMOTE_CODE \
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