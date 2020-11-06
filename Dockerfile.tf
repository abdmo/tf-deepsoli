#FROM tensorflow/tensorflow:latest-gpu
FROM tensorflow/tensorflow:1.15.4-gpu-py3

RUN apt-get update && apt-get install -y \
    vim

WORKDIR /workdir
