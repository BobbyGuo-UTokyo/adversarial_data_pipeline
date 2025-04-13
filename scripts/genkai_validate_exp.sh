#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L elapse=12:00:00
#PJM -L gpu=1
#PJM -N name=adversarial_1
#PJM -j

# in case 'module' command is not foundd
source /etc/profile.d/modules.sh
module load cuda/12.2.2
module load gcc-toolset/12
source /home/pj24002027/ku40003401/python_env/adversarial/bin/activate

cd /home/pj24002027/ku40003401/repos/adversarial_data_pipeline
# register huggingface, modify before submitting
export HF_TOKEN=XXX
huggingface-cli login --token $HF_TOKEN
# For vllm, set GPUs
export CUDA_VISIBLE_DEVICES=0

# Scripts
python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" \
    --output_dir test_adversarial_results/math_0411/original_no_train \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_original_pos_original_neg" \
    --output_dir test_adversarial_results/math_0411/sft_original_pos_original_neg \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_original_pos_original_neg_continue_pretrain" \
    --output_dir test_adversarial_results/math_0411/sft_original_pos_original_neg_continue_pretrain \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_original_pos_adversarial_neg" \
    --output_dir test_adversarial_results/math_0411/sft_original_pos_adversarial_neg \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_original_pos_adversarial_neg_continue_pretrain" \
    --output_dir test_adversarial_results/math_0411/sft_original_pos_adversarial_neg_continue_pretrain \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_adversarial_pos_original_neg" \
    --output_dir test_adversarial_results/math_0411/sft_adversarial_pos_original_neg \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_adversarial_pos_original_neg_continue_pretrain" \
    --output_dir test_adversarial_results/math_0411/sft_adversarial_pos_original_neg_continue_pretrain \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_adversarial_pos_adversarial_neg" \
    --output_dir test_adversarial_results/math_0411/sft_adversarial_pos_adversarial_neg \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_adversarial_pos_adversarial_neg_continue_pretrain" \
    --output_dir test_adversarial_results/math_0411/sft_adversarial_pos_adversarial_neg_continue_pretrain \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_adversarial_neg" \
    --output_dir test_adversarial_results/math_0411/sft_adversarial_neg \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm

python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "/home/pj24002027/ku40003401/repos/OpenRLHF/outputs/sft_results/math/Qwen2.5-Math-7B-Instruct/sft_adversarial_neg_continue_pretrain" \
    --output_dir test_adversarial_results/math_0411/sft_adversarial_neg_continue_pretrain \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm
