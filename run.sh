#!bin/bash

# NOTE: env vars are now passed via docker run --env-file
#set -a # automatically export all variables
#source config.env 
#set +a

set -e # exit on error

if [[ -z "${HUGGINGFACEHUB_API_TOKEN}" ]]; then
  echo "HUGGINGFACEHUB_API_TOKEN not found. Please set it in .env file."
  exit 1
fi

if [[ -z "${GPU_MEMORY_UTILIZATION}" ]]; then
  gpu_memory_utilization=0.9
else
  gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
fi

if [[ -z "${VLLM_SWAP_SPACE}" ]]; then
  swap_space=4
else
  swap_space=$VLLM_SWAP_SPACE
fi

# Local TMPPATHS to store intermediate files

if [[ -z "${COTEVAL_CACHE_DIR}" ]]; then
  COTEVAL_CACHE_DIR="./cot-eval-cache"
fi

LOTMP_NEXTMODELINFO="$COTEVAL_CACHE_DIR/next_model.json"  # stores info about which model to evaluate next
LOTMP_CONFIGKEYSINFO="$COTEVAL_CACHE_DIR/config_keys.txt"  # stores names of cot-eval configs that will be used
LOTMP_CONFIGSFOLDER="$COTEVAL_CACHE_DIR/cot_eval_configs"  # folder with cot-eval configs that will be used
LOTMP_ELEU_CONFIGSFOLDER="$COTEVAL_CACHE_DIR/eleuther/tasks/logikon"  # folder with lm-eval-harness tasks
LOTMP_ELEU_CONFIGSINFO="$COTEVAL_CACHE_DIR/lm_eval_harness_tasks.json"  # groups names of lm-eval-harness tasks that will be used
LOTMP_ELEU_OUTPUTDIR="$COTEVAL_CACHE_DIR/eleuther/output"  # folder with lm-eval-harness output

# cp pre-built eleuther tasks and templates to cache dir
cp -r ./eleuther/tasks/logikon/* $LOTMP_ELEU_CONFIGSFOLDER

##############################
# login to huggingface hub

huggingface-cli login --token $HUGGINGFACEHUB_API_TOKEN


##############################
# lookup model to-be evaluated

if [[ -z "${NEXT_MODEL_PATH}" ]]; then
  python scripts/lookup_pending_model.py --keys_file $LOTMP_NEXTMODELINFO --max_params $MAX_MODEL_PARAMS --requests_repo $REQUESTS_REPO
  model=$(cat $LOTMP_NEXTMODELINFO | jq -r .model)
  revision=$(cat $LOTMP_NEXTMODELINFO | jq -r .revision)
  precision=$(cat $LOTMP_NEXTMODELINFO | jq -r .precision)
else
  model="${NEXT_MODEL_PATH}"
  revision="${NEXT_MODEL_REVISION}"
  precision="${NEXT_MODEL_PRECISION}"
fi
echo "Model to evaluate: $model : $revision. Precision: $precision"

# set lm-eval-harness model_args
lm_eval_model_args="pretrained=${model},revision=${revision},dtype=${precision},tensor_parallel_size=${NUM_GPUS},gpu_memory_utilization=${gpu_memory_utilization},trust_remote_code=$TRUST_REMOTE_CODE"
if [[ -z "${MAX_LENGTH}" ]]; then
  echo "No MAX_LENGTH specified in config."
  cot_config_extra_args=""
else
  lm_eval_model_args="${lm_eval_model_args},max_length=$MAX_LENGTH"
  cot_config_extra_args="--max_model_len $MAX_LENGTH"
fi
echo "lm-eval model_args: $lm_eval_model_args"


##############################
# create CoT configs
# a 'config' defines how reasoning traces are generated for a given task
python scripts/create_cot_configs.py $cot_config_extra_args \
    --model $model \
    --revision $revision \
    --precision ${precision} \
    --chains $CHAINS \
    --model_kwargs "$MODELKWARGS" \
    --tasks $TASKS \
    --output_dir $LOTMP_CONFIGSFOLDER \
    --template_path "./src/cot_eval/configs/template.yaml" \
    --keys_file $LOTMP_CONFIGKEYSINFO \
    --num_gpus $NUM_GPUS \
    --swap_space $swap_space
configkeys=$(cat $LOTMP_CONFIGKEYSINFO)  # format is "config1,config2,config3"
echo "Created configs: $configkeys and stored in $LOTMP_CONFIGSFOLDER"


##############################
# generate reasoning traces
# run cot_eval to create reasoning traces for every config (model and task)
# reasoning traces are uploaded to huggingface hub
arr_configkeys=(${configkeys//,/ })
for config in "${arr_configkeys[@]}"
do
    cot-eval \
        --config "${LOTMP_CONFIGSFOLDER}/${config}.yaml" \
        --upload_dataset $TRACES_REPO \
        --hftoken $HUGGINGFACEHUB_API_TOKEN
done


##############################
# create lm-eval-harness tasks
# a 'harness task' defines how to evaluate a given model on a given task,
# specifically whether to include the model's reasoning traces or not
python scripts/create_lm_eval_harness_tasks.py \
    --configs $configkeys \
    --output_dir $LOTMP_ELEU_CONFIGSFOLDER \
    --configs_dir $LOTMP_CONFIGSFOLDER \
    --traces_dataset_path $TRACES_REPO \
    --keys_file $LOTMP_ELEU_CONFIGSINFO
harness_tasks_base=$(cat $LOTMP_ELEU_CONFIGSINFO | jq -r .base) # format is "task1,task2,task3"
harness_tasks_cot=$(cat $LOTMP_ELEU_CONFIGSINFO | jq -r .cot) # format is "task1,task2,task3"
echo "Created lm-eval-harness tasks base: $harness_tasks_base" # no cot
echo "Created lm-eval-harness tasks cot: $harness_tasks_cot"


timestamp=$(date +"%y-%m-%d-%T")

##############################
# ORIG evaluation
# run lm-eval originial BASE (unperturbed) for each task
if [ "$DO_BASEEVAL" = true ] ; then
    arrTASKS=(${TASKS//,/ })
    basetasks=$(printf "%s_base," "${arrTASKS[@]}")
    basetasks=${basetasks:0:-1}
    output_path=$LOTMP_ELEU_OUTPUTDIR/${model}/orig/results_${timestamp}.json
    if [ -f $output_path ]; then
        echo "Outputfile $FILE exists. Skipping eval of $basetasks."
    else
        lm-eval --model vllm \
            --model_args $lm_eval_model_args \
            --tasks $basetasks \
            --num_fewshot 0 \
            --batch_size auto \
            --output_path $output_path \
            --include_path $LOTMP_ELEU_CONFIGSFOLDER
    fi
fi


##############################
# BASE and COT evaluation
# run lm evaluation harness for each of the tasks

# without reasoning traces
lm-eval --model vllm \
    --model_args $lm_eval_model_args \
    --tasks ${harness_tasks_base} \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path $LOTMP_ELEU_OUTPUTDIR/${model}/base/${timestamp}.json \
    --include_path $LOTMP_ELEU_CONFIGSFOLDER

# with reasoning traces
arrHT=(${harness_tasks_cot//,/ })
ht_batch_size=5
# batched processing of harness_tasks_cot
for((i=0; i < ${#arrHT[@]}; i+=ht_batch_size))
do
    ht_batch=( "${arrHT[@]:i:ht_batch_size}" )
    ht_batch_s=$(printf ",%s" "${ht_batch[@]}")
    ht_batch_s=${ht_batch_s:1}
    echo "Evaluating cot tasks: $ht_batch_s"

    lm-eval --model vllm \
        --model_args $lm_eval_model_args \
        --tasks ${ht_batch_s} \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path $LOTMP_ELEU_OUTPUTDIR/${model}/cot/${timestamp}_idx${i}.json \
        --include_path $LOTMP_ELEU_CONFIGSFOLDER
done

##############################
# collect and upload results
python scripts/upload_results.py \
    --model $model \
    --revision $revision \
    --precision $precision \
    --tasks $TASKS \
    --timestamp $timestamp \
    --output_dir $LOTMP_ELEU_OUTPUTDIR \
    --results_repo $RESULTS_REPO \
    --requests_repo $REQUESTS_REPO \
    --leaderboard_results_repo $LEADERBOARD_RESULTS_REPO \
    --create_pr $CREATE_PULLREQUESTS

