#!/bin/sh
export OPENCL_PLATFORM='NVIDIA CUDA'
export OPENCL_DEVICE='Tesla M60'
export OPENCL_SOURCE=nvidia.cl
export OPENCL_WORKGROUP_OFFSET='0 0 0'
export OPENCL_WORKGROUP_GLOBAL='256 112 112'

set -x
for x in 1 2 4 8 16
do
    for y in 1 2 4
    do
        for z in 1 2 4 7 8 14 16 28
        do
            export OPENCL_WORKGROUP_LOCAL="$x $y $z"
            ./cnn
        done
    done
done

