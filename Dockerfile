FROM bitnamilegacy/spark:3.5.1-debian-12-r10
# FROM python:3.11-slim

# set working directory
WORKDIR /app

#copy dataset to docker environment
COPY src/dataset/medical_transcriptions.csv /app/dataset/

# ENV JAVA_HOME = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
# ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Custom logging
# COPY log4j2.properties /opt/bitnami/spark/conf/log4j2.properties

# copy requirements file and install python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# set global python path
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Set a writable cache directory for huggingface transformers
# ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# any files/libraries you need on the cluster, install here ie:
# RUN pip install scipy