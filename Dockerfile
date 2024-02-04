FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL maintainer "Gregor Betz and the Logikon AI Team"

ARG VLLM_VERSION=0.2.7

# Copy local config file to the container image.

ENV APP_HOME . 
WORKDIR $APP_HOME
COPY .config.env ./config.env

# Clone repos

RUN git clone https://github.com/logikon-ai/cot-eval.git
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness.git

# Install python packages

RUN pip install --upgrade pip
RUN pip uninstall transformer-engine -y

RUN cd lm-evaluation-harness && pip install -e ".[vllm]"
RUN cd cot-eval && pip install -e ".[cuda]"

RUN pip install -U vllm==${VLLM_VERSION}

# Run cot-eval script on startup

WORKDIR ${APP_HOME}/cot-eval
CMD ["bash", "run.sh"]
