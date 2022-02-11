#!/bin/bash

#block(name=exp1, threads=5, memory=10000, subtasks=1, gpu=true, hours=666)
CUDA_VISIBLE_DEVICES=0 python -u base.py
