FROM --platform=linux/amd64 lmsysorg/sglang:v0.5.8.post1-runtime

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY . /workspace/MMIRAGE
WORKDIR /workspace/MMIRAGE

RUN pip install --no-cache-dir .

ENTRYPOINT ["python3", "-m", "mmirage.shard_process"]
