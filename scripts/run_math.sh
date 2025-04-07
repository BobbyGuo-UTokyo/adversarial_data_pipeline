python3 math_pipeline/generative_adversarial.py --data_names gsm8k,math --data_dir data/ \
    --model_name_or_path VolcEngine_DeepSeekR1 \
    --output_dir ./output_adversarial --prompt_type qwen25-math-cot --split train \
    --granularity_prompt "Mainly modify variable names like item names or person names and try to keep other things unchanged" \
    --num_test_sample 20 --seed 42 --temperature 0 --shuffle