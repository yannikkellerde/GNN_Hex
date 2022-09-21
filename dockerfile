FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
RUN apt-get update

ENV TORCH_CUDA_ARCH_LIST=Volta;Turing
RUN apt-get install -y git build-essential curl wget python3.8 python3-setuptools python3.8-dev
WORKDIR /root
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y python3-graph-tool
RUN apt-get install -y pkg-config libcairo2-dev libjpeg-dev libgif-dev
RUN python3.8 -m pip install matplotlib pycairo numpy scipy alive-progress wandb rich blessings
RUN python3.8 -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.8 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
COPY .ssh/ ./.ssh/
RUN git clone git@github.com:yannikkellerde/Gabor_Graph_Networks.git
WORKDIR /root/Gabor_Graph_Networks/GN0/
RUN git clone git@github.com:yannikkellerde/Rainbow.git
WORKDIR /root/Gabor_Graph_Networks/
RUN python3.8 -m pip install -e .
WORKDIR /root/
COPY .secrets/api_key .
RUN wandb login $(cat api_key)
