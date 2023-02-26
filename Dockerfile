FROM nytimes/blender:3.3.1-gpu-ubuntu18.04 AS blender
# install git and pull the blender generation script
RUN apt-get update && apt-get install -y git python3.10

WORKDIR /workspace/blender_gen


COPY requirements.txt .
RUN pip3.10 install -r requirements.txt

COPY src ./src
COPY main.py .
RUN mkdir -p /data

RUN echo built

# run the blender generation script
ENTRYPOINT ["python3.10", "main.py"]