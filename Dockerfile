FROM ubuntu:20.04

# tzdata対策環境変数
ENV DEBIAN_FRONTEND=noninteractive

# ラベル
# LABEL maintainer="oshimamasara@gmail.com"
LABEL version="1.0"
LABEL description="動画から動く人体検知動画の作成"


# aptの更新
RUN apt update && \
    apt upgrade 

# Pythonの前提ライブラリ
RUN apt install -y wget && \
    apt install -y build-essential checkinstall && \
    apt install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
    apt install -y git

# Python
RUN wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tar.xz && \
    tar xvf Python-3.6.9.tar.xz && \
    cd Python-3.6.9 && \
    ./configure && \
    make altinstall

# TensorFlow、OpenPoseの前提ライブラリ
RUN pip3.6 install --upgrade pip && \
    pip3.6 install opencv-contrib-python && \
    pip3.6 install opencv-python && \
    pip3.6 install tensorflow-gpu==1.14 && \
    pip3.6 install numpy && \
    pip3.6 install --upgrade cython && \
    apt -yV install libopenblas-dev && \
    apt -yV install liblapacke-dev && \
    apt -yV install libopenblas-dev && \
    apt -yV install gfortran && \
    apt -y install swig

# TensorFlow、OpenPose
RUN git clone https://github.com/ildoonet/tf-pose-estimation.git && \
    cd tf-pose-estimation/ && \
    pip3.6 install -r requirements.txt && \
    cd tf_pose/pafprocess && \
    swig -python -c++ pafprocess.i && python3.6 setup.py build_ext --inplace && \
    cd ../../ && \
    python3.6 setup.py install
