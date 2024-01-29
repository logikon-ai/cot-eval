#!bin/bash

CHAINS="HandsOn,ReflectBeforeRun" # "HandsOn"
MODELKWARGS='[{temperature: .3, top_k: 100, top_p: .95},{temperature: 0},{temperature: 0, use_beam_search: true, best_of: 2, n: 1}]'  # YAML format
TASKS="logiqa,logiqa2,lsat-ar,lsat-rc,lsat-lr"
OUTPUT_DIR="./eleuther/output"
CONFIGS_DIR="src/cot_eval/configs"
TRUST_REMOTE_CODE=true
MAX_LENGTH=4096
DO_BASEEVAL=true
MAX_MODEL_PARAMS=10 # max number of params (B) of evaluated model

set -a # automatically export all variables
source ../.env
set +a

set -e # exit on error

if [[ -z "${HUGGINGFACEHUB_API_TOKEN}" ]]; then
  echo "HUGGINGFACEHUB_API_TOKEN not found. Please set it in .env file."
  exit 1
fi


huggingface-cli login --token $HUGGINGFACEHUB_API_TOKEN

# lookup model to-be evaluated
python scripts/lookup_pending_model.py --keys_file ./next_model.json --max_params $MAX_MODEL_PARAMS
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

