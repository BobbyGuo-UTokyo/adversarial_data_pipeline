python3 math_pipeline/generative_adversarial.py --data_names gsm8k,math --data_dir data/ \
    --model_name_or_path VolcEngine_DeepSeekR1 \
    --output_dir ./output_adversarial --prompt_type qwen25-math-cot --split train \
    --prompt_file prompt_overall_level.txt \
    --num_test_sample 100 --seed 666 --temperature 0 --shuffle --use_math_verify --postfix "overall_level"