FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y python3 python3-pip

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools

RUN python3 -m pip install --no-cache-dir --upgrade fschat[model_worker]
