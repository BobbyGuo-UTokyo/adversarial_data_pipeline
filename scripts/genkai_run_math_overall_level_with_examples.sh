#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L elapse=72:00:00
#PJM -L gpu=1
#PJM -N name=adversarial_1
#PJM -j


module load cuda/12.6.1
# module load cudnn/8.9.7
module load gcc-toolset/13
# module load nccl/2.22.3
source /home/pj24002027/ku40003401/python_env/adversarial/bin/activate

cd /home/pj24002027/ku40003401/repos/adversarial_data_pipeline

python3 math_pipeline/generative_adversarial.py --data_names gsm8k,math --data_dir data/ \
    --model_name_or_path VolcEngine_DeepSeekR1 --endpoint_id ep-20250216235228-69vhs \
    --output_dir ./output_adversarial --prompt_type qwen25-math-cot --split train \
    --prompt_file prompts/prompt_overall_level_with_examples.txt \
    --num_test_sample 100 --seed 888 --temperature 0 --shuffle --use_math_verify --postfix "overall_level_with_examples"