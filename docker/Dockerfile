FROM lmsysorg/sglang:v0.5.3rc1-cu126

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY . /workspace/MMIRAGE
WORKDIR /workspace/MMIRAGE

RUN pip install --no-cache-dir .[gpu]

ENTRYPOINT ["/bin/bash"]
