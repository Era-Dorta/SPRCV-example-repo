FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

COPY requirements.txt .

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y git unzip wget && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install -r requirements.txt && \
    pip cache purge