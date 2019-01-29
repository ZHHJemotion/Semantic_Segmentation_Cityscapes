#!/usr/bin/env bash

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

log_dir=/home/pingguo/PycharmProject/dl_project/Weights/PSnet/logs/PS-Net-2018-08-01-20-03-05/
tb_dir=/usr/local/lib/python3.5/dist-packages/tensorboard/
python3 ${tb_dir}main.py --logdir=${log_dir} --port=9009
