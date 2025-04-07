python3 math_pipeline/generative_adversarial.py --data_names gsm8k,math --data_dir data/ \
    --model_name_or_path VolcEngine_DeepSeekR1 \
    --output_dir ./output_adversarial --prompt_type qwen25-math-cot --split train \
    --granularity_prompt "The problem setup should be in a different scenario, while the wording and structure of sentences should be altered. The intermediate variable values can be different." \
    --num_test_sample 100 --seed 321 --temperature 0 --shuffle --use_math_verify --postfix "full_change"