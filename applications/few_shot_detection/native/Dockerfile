ARG BASE_IMAGE=advigw/isaac-ros-base-jazzy:1.00
FROM ${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y cmake g++ && rm -rf /var/lib/apt/lists/*
COPY . /tmp/few-shot/native/
RUN bash /tmp/few-shot/native/build.sh gpu_geometry \
    && cmake --install /tmp/few-shot/build/native --prefix /usr/local \
    && ldconfig && rm -rf /tmp/few-shot
