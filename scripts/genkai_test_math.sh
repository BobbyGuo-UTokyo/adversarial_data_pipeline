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

export CUDA_VISIBLE_DEVICES=0,1

python3 math_pipeline/math_eval.py --data_names math,gsm8k --data_dir outputs/adversarial_attack \
    --adversarial_generation_file_name train_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" --output_dir test_adversarial_results \
    --output_name train_qwen25-math-cot_-1_seed0_t0.0_s0_e-1_test_result.jsonl \
    --prompt_type qwen25-math-cot --split train --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --test_adversarial_generation --save_outputs --use_safetensors --use_vllm
