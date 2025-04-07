python3 math_pipeline/generative_adversarial.py --data_names gsm8k,math --data_dir data/ \
    --model_name_or_path VolcEngine_DeepSeekR1 \
    --output_dir ./output_adversarial --prompt_type qwen25-math-cot --split train \
    --granularity_prompt "The problem should be paraphrased without changing variable names like item names or person names, while variable values must not change" \
    --num_test_sample 100 --seed 321 --temperature 0 --shuffle --postfix "paraphrase"