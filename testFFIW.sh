#!/bin/bash

low[0]='low'
low[1]='medium'

for ((j=0;j<2;j++))
do
    CUDA_VISIBLE_DEVICES=1 python inference.py --low ${low[j]}
done