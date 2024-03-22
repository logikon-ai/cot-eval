FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL maintainer "Gregor Betz and the Logikon AI Team"

ARG VLLM_VERSION=0.3.2
ARG LM_EVAL_VERSION=v0.4.1

ENV APP_HOME . 
WORKDIR $APP_HOME

# Clone repos

RUN git clone  https://github.com/logikon-ai/cot-eval.git
RUN git clone --branch ${LM_EVAL_VERSION} https://github.com/EleutherAI/lm-evaluation-harness.git

# Install python packages

RUN pip install --upgrade pip
RUN pip uninstall transformer-engine -y

RUN cd lm-evaluation-harness && pip install -e ".[vllm]"
RUN cd cot-eval && pip install -e .
RUN pip install -U vllm==${VLLM_VERSION}


# Environment variables

ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Run cot-eval script on startup

WORKDIR ${APP_HOME}/cot-eval
CMD ["bash", "run.sh"]
