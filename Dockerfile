FROM nvcr.io/nvidia/pytorch:24.05-py3

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

RUN cd lm-evaluation-harness && pip install -e .
RUN cd cot-eval && pip install -e .
RUN pip install -U vllm==${VLLM_VERSION} mistral-common==1.4.0

# Install datasets 2.18.0, being used with lm-evaluation-harness
RUN pip install datasets>=2.18.0

# Uninstall pynvml and install nvidia-ml-py instead
RUN pip uninstall -y pynvml
RUN pip install nvidia-ml-py

# Reinstall flash-attn as torch might have gotten reinstalled above
RUN pip uninstall -y flash-attn
RUN pip install flash-attn --no-build-isolation

# Install flashinfer backend
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
# Run cot-eval script on startup

WORKDIR ${APP_HOME}/cot-eval
CMD ["bash", "run.sh"]
