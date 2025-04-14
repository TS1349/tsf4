#!/bin/bash

#OAR -p gpu='YES'

#OAR -l /gpunum=1, walltime=1

#OAR -t besteffort

#OAR --name test

module load cuda/12.2 cudnn/8.9-cuda-12.1

source activate tsf4

python ./test.py
