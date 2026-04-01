FROM lmsysorg/sglang:latest

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY . /workspace/MMIRAGE
WORKDIR /workspace/MMIRAGE

RUN pip install --no-cache-dir .

ENTRYPOINT ["python3", "-c", "import mmirage.core.process, runpy; runpy.run_module('mmirage.shard_process', run_name='__main__')"]
