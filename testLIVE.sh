#!/bin/bash

# method[0]='Before'
# method[1]='After'
# method[2]='Stream'

# for ((j=0;j<3;j++))
# do
#     CUDA_VISIBLE_DEVICES=1 python inference.py --method ${method[j]}
# done

CUDA_VISIBLE_DEVICES=1 python inference.py --method Before