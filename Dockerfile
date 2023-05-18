FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

# Python 3.8 and pip3
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python3.8
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt-get -y install python3-pip \
                       libpython3.8 \
                       python3.8-distutils \
                       ffmpeg libsm6 libxext6

RUN python -m pip --version

COPY . /app
WORKDIR /app
RUN python -m pip install cmake
RUN python -m pip install -r requirements.txt

CMD PYTHONPATH=. python main/main.py