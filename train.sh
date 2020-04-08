#!/usr/bin/env sh


export PATH=${PATH}:/home/gp/caffe_ssd/build/tools
export PYTHONPATH=/home/gp/caffe_ssd/python:${PYTHONPATH}

caffe train -solver="solver_train.prototxt" -weights="mobilenet_iter_73000.caffemodel" -gpu 0  2>&1|tee train_20200401.log
