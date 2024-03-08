FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL maintainer "Gregor Betz and the Logikon AI Team"

ARG VLLM_VERSION=0.3.3
ARG LM_EVAL_VERSION=0.4.0

ENV APP_HOME . 
WORKDIR $APP_HOME

# Clone repos

RUN git clone  https://github.com/logikon-ai/cot-eval.git
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness.git@${LM_EVAL_VERSION}

# Install python packages

RUN pip install --upgrade pip
RUN pip uninstall transformer-engine -y

RUN cd lm-evaluation-harness && pip install -e ".[vllm]"
RUN cd cot-eval && pip install -e .
RUN pip install -U vllm==${VLLM_VERSION}


# Run cot-eval script on startup

WORKDIR ${APP_HOME}/cot-eval
CMD ["bash", "run.sh"]
