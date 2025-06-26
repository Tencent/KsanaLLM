#!/bin/bash

SCRIPT_HOME=`pwd`

cd 3rdparty/LLM_kernels/csrc/kernels/nvidia/fused_moe
bash fused_moe_creator.sh `pwd` $SCRIPT_HOME/build/triton_kernel_files

cd $SCRIPT_HOME

cd 3rdparty/LLM_kernels/csrc/kernels/nvidia/decode_grouped_att_m_fwd
bash decode_grouped_att_m_fwd_creator.sh `pwd` $SCRIPT_HOME/build/triton_kernel_files

cd $SCRIPT_HOME

cd 3rdparty/LLM_kernels/csrc/kernels/nvidia/decode_softmax_reducev_fwd
bash decode_softmax_reducev_fwd_creator.sh `pwd` $SCRIPT_HOME/build/triton_kernel_files

cd $SCRIPT_HOME

cd 3rdparty/LLM_kernels/csrc/kernels/nvidia/per_token_group_quant
bash per_token_group_quant_creator.sh `pwd` $SCRIPT_HOME/build/triton_kernel_files

cd $SCRIPT_HOME

cd 3rdparty/LLM_kernels/csrc/kernels/nvidia/fused_moe_kernel_gptq_awq
bash fused_moe_kernel_gptq_awq_creator.sh `pwd` $SCRIPT_HOME/build/triton_kernel_files

cd $SCRIPT_HOME

cd 3rdparty/LLM_kernels/csrc/kernels/nvidia/fused_moe_gptq_int4_fp8_kernel
bash fused_moe_gptq_int4_fp8_kernel_creator.sh `pwd` $SCRIPT_HOME/build/triton_kernel_files
