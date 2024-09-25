#!bin/bash

set -a # automatically export all variables
source config.env 
# source .venv-cot-eval/bin/activate
set +a

set -e # exit on error

if [[ ${VENV_DIR} ]]; then
  source "${VENV_DIR}/bin/activate"
fi

if [[ ${PYTHON_COMMAND} ]]; then
  alias python="${PYTHON_COMMAND}"
fi
python --version



if [[ -z "${HUGGINGFACEHUB_API_TOKEN}" ]]; then
  echo "HUGGINGFACEHUB_API_TOKEN not found. Please set it in .env file."
  exit 1
fi

if ${COT_EVAL_DEBUG}; then
  TRACES_REPO="${TRACES_REPO_DEBUG}"
  CREATE_PULLREQUESTS="true"
  LM_EVAL_VERBOSITY="DEBUG"
  LM_EVAL_LIMIT_ARG="--limit 10 "
  echo "INFO: Debug mode enabled! Using debug traces repo: $TRACES_REPO. Creating pull requests."
else 
  LM_EVAL_VERBOSITY="INFO"
  LM_EVAL_LIMIT_ARG=""
fi

if [[ -z "${INFERENCE_BASE_URL}" ]]; then
  echo "No INFERENCE_BASE_URL specified in config, defaulting to localhost:8000."
  base_url="http://localhost:8000/v1"
else
  base_url="${INFERENCE_BASE_URL}"
fi

if [[ -z "${INFERENCE_BATCH_SIZE}" ]]; then
  batch_size="1"
else
  echo "Warning: INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE} from config will be ignored. Harness multiple choice tasks are evaluated with batch_size=1."
  batch_size="1"
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
LOTMP_DEFAULT="$COTEVAL_CACHE_DIR/TMP"  # folder with other temporary files

# cp pre-built eleuther tasks and templates to cache dir
mkdir -p $LOTMP_DEFAULT
mkdir -p $LOTMP_ELEU_CONFIGSFOLDER
cp -r ./eleuther/tasks/logikon/* $LOTMP_ELEU_CONFIGSFOLDER

##############################
# login to huggingface hub

huggingface-cli login --token $HUGGINGFACEHUB_API_TOKEN


##############################
# lookup model to-be evaluated

if [[ -z "${NEXT_MODEL_PATH}" ]]; then
  python scripts/lookup_pending_model.py --keys_file $LOTMP_NEXTMODELINFO --max_params $MAX_MODEL_PARAMS --requests_repo $REQUESTS_REPO --tmp_dir $LOTMP_DEFAULT
  model=$(cat $LOTMP_NEXTMODELINFO | jq -r .model)
  revision=$(cat $LOTMP_NEXTMODELINFO | jq -r .revision)
  precision=$(cat $LOTMP_NEXTMODELINFO | jq -r .precision)
else
  model="${NEXT_MODEL_PATH}"
  revision="${NEXT_MODEL_REVISION}"
  precision="${NEXT_MODEL_PRECISION}"
fi
echo "Model to evaluate: $model : $revision. Precision: $precision"

# cot_config_extra_args
if [[ -z "${MAX_LENGTH}" ]]; then
  echo "No MAX_LENGTH specified in config."
  cot_config_extra_args=""
else
  cot_config_extra_args="--max_model_len $MAX_LENGTH"
fi

# set lm-eval-harness model_args
lm_eval_model_args="base_url=${base_url}/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,model=${model}"

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
    --keys_file $LOTMP_CONFIGKEYSINFO
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
        --base_url $base_url \
        --upload_dataset $TRACES_REPO \
        --hftoken $HUGGINGFACEHUB_API_TOKEN \
        --debug $COT_EVAL_DEBUG
done


##############################
# create lm-eval-harness tasks
# a 'harness task' defines how to evaluate a given model on a given task,
# specifically whether to include the model's reasoning traces or not
python scripts/create_lm_eval_harness_tasks.py \
    --model $model \
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
    output_path=$LOTMP_ELEU_OUTPUTDIR/${model}/orig/results_${timestamp}
    if [ -f $output_path ]; then
        echo "Outputfile $FILE exists. Skipping eval of $basetasks."
    else
        lm-eval --model local-completions ${LM_EVAL_LIMIT_ARG} \
            --model_args $lm_eval_model_args \
            --batch_size $batch_size \
            --tasks $basetasks \
            --num_fewshot 0 \
            --output_path $output_path \
            --include_path $LOTMP_ELEU_CONFIGSFOLDER \
            --verbosity $LM_EVAL_VERBOSITY
    fi
fi


##############################
# BASE and COT evaluation
# run lm evaluation harness for each of the tasks

# without reasoning traces
lm-eval --model local-completions \
    --model_args $lm_eval_model_args \
    --batch_size $batch_size \
    --tasks ${harness_tasks_base} \
    --num_fewshot 0 \
    --output_path $LOTMP_ELEU_OUTPUTDIR/${model}/base/${timestamp} \
    --include_path $LOTMP_ELEU_CONFIGSFOLDER \
    --verbosity $LM_EVAL_VERBOSITY

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

    lm-eval --model local-completions \
        --model_args $lm_eval_model_args \
        --batch_size $batch_size \
        --tasks ${ht_batch_s} \
        --num_fewshot 0 \
        --output_path $LOTMP_ELEU_OUTPUTDIR/${model}/cot/${timestamp}_idx${i} \
        --include_path $LOTMP_ELEU_CONFIGSFOLDER \
        --verbosity $LM_EVAL_VERBOSITY
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
    --tmp_dir $LOTMP_DEFAULT \
    --results_repo $RESULTS_REPO \
    --requests_repo $REQUESTS_REPO \
    --leaderboard_results_repo $LEADERBOARD_RESULTS_REPO \
    --create_pr $CREATE_PULLREQUESTS


# deactivate
# unalias python
