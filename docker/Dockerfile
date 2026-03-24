FROM docker.io/lmsysorg/sglang:v0.5.8.post1

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY . /workspace/MMIRAGE
WORKDIR /workspace/MMIRAGE

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "mmirage.shard_process"]
