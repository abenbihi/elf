#FROM ubuntu:16.04
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

############################################# Proxy definition
# This is necessary and specific to CentraleSupelec
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV no_proxy ${no_proxy}

# Setup User to match Host User, and give superuser permissions
# 
ARG USER_ID=0
ARG GROUP_ID=0
RUN groupadd -g ${GROUP_ID} code_executor && useradd -m code_executor -u ${USER_ID} -g ${GROUP_ID}
###########################

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost1.58-all-dev \
    libopencv-dev \
    libopenmpi-dev \
    python-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-setuptools \
    python3-scipy \
    vim \
    wget \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*


ENV WS /home/ws
WORKDIR $WS


#ENV CODE_DIR /root/code
#ENV VENV /root/venv
#
#RUN \
#    pip install virtualenv && \
#    virtualenv $VENV --python=python3 && \
#    . $VENV/bin/activate && \
#    pip install --upgrade pip

#ENV PATH=$VENV/bin:$PATH
ENV PATH=$WS/bin:$PATH

COPY . $WS
#WORKDIR $CODE_DIR

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip3 install --upgrade pip==9.0.1 && \
    for req in $(cat requirements.txt) pydot; do pip3 install $req; done 
#&& \
#    pip install -e .[test]

ENV CUDA_VISIBLE_DEVICES 0
#ENV PYTHONPATH $CODE_DIR:$CODE_DIR/slim:$PYTHONPATH
ENV PYTHONPATH $WS:$PYTHONPATH
CMD /bin/bash
