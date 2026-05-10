FROM lmsysorg/sglang:latest

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY . /workspace/MMIRAGE
WORKDIR /workspace/MMIRAGE

RUN pip install --no-cache-dir .[gpu]

ENTRYPOINT ["python3", "-m", "mmirage.shard_process"]
