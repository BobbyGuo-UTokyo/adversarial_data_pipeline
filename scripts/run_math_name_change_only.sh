python3 math_pipeline/generative_adversarial.py --data_names gsm8k,math --data_dir data/ \
    --model_name_or_path VolcEngine_DeepSeekR1 --endpoint_id ep-20250216235228-69vhs \
    --output_dir ./output_adversarial --prompt_type qwen25-math-cot --split train \
    --granularity_prompt "Only modify variable names like item names or person names and strictly keep other things unchanged, including variable values" \
    --num_test_sample 100 --seed 321 --temperature 0 --shuffle --postfix "name_change_only"