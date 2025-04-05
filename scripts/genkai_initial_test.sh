#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L elapse=12:00:00
#PJM -L gpu=2
#PJM -N name=adversarial_1
#PJM -j


module load cuda/12.6.1
# module load cudnn/8.9.7
module load gcc-toolset/13
# module load nccl/2.22.3
source /home/pj24002027/ku40003401/python_env/adversarial/bin/activate

cd /home/pj24002027/ku40003401/repos/adversarial_data_pipeline
# register huggingface, modify before submitting
export HF_TOKEN=XXX
huggingface-cli login --token $HF_TOKEN
# For vllm, set GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Scripts
# sh scripts/initial_test_qwen25_math_7b.sh
# sh scripts/initial_test_qwen25_math_1_5b.sh
sh scripts/initial_test_llama_31_8b.sh
# sh scripts/initial_test_finemath_llama_3b.sh
# sh scripts/initial_test_mathstral_7b_v01.sh
sh scripts/initial_test_gemma3_1b.sh
sh scripts/initial_test_gemma3_4b.sh
sh scripts/initial_test_gemma3_12b.sh
