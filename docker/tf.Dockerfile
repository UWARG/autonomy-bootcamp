FROM tensorflow/tensorflow:devel-gpu
RUN pip3 install \
    tensorflow \
    tensorflow-datasets \
    pyyaml

WORKDIR /root/src