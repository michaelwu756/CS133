# Copyright (C) 2019 Falcon Computing Solutions, Inc. - All rights reserved.
#
# Choose target FPGA platform & vendor
VENDOR=XILINX
DEVICE=xilinx_aws-vu9p-f1-04261818_dynamic_5_0
#DEVICE=xilinx:adm-pcie-ku3:2ddr:3.3
#DEVICE=xilinx:adm-pcie-ku3:2ddr-xpr:4.0
#DEVICE=xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0

# Edit Options per your setup if using AWS F1
#S3_OPT=-s3_bucket=fcs-fpga -s3_dcp_key=dcp -s3_logs_key=log

# Host Code Compilation settings
SRC_DIR:=../../src
INC_DIR:=../../include
COMMON_DIR:=../../common
HOST_SRC_FILES=$(SRC_DIR)/main.cpp $(SRC_DIR)/cnn.cpp
DATA_DIR=../../data
F1_DATA_DIR=./data

# Executable names and arguments
EXE=cnn
ACC_EXE=$(EXE)_acc

# Testing mode
EXE_ARGS=../../../ 
F1_EXE_ARGS=../../../


CXX_INC_DIRS=-I . -I $(INC_DIR)
CXX_FLAGS+= $(CXX_INC_DIRS)
CXX_FLAGS+= -Wall -O3 -lrt -std=c++11
CXX_FLAGS+=-DPLATFORM=\"$(PLATFORM)\" -DKERNEL=\"$(KERNEL_NAME)\" -DKERNEL_NAME=$(KERNEL_NAME)
#CXX_FLAGS+= -g

# Accelerated Kernel settings
KERNEL_NAME=CnnKernel
KERNEL_SRC_FILES=$(SRC_DIR)/CnnKernel.cpp
KERNEL_INC_DIR=$(CXX_INC_DIRS)

# MerlinCC Options
CMP_OPT= --attribute auto_func_inline=aggressive
LNK_OPT= 

#common make
MCC_COMMON_DIR=../../../common
include $(MCC_COMMON_DIR)/mcc_common.mk

