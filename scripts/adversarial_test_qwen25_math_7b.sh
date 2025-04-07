python3 math_pipeline/generative_adversarial.py --data_names gsm8k,math --data_dir outputs/initial_inference \
    --attacker_model_name_or_path VolcEngine_DeepSeekR1 \
    --original_model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" \
    --output_dir outputs/adversarial_attack --prompt_type qwen25-math-cot --split train \
    --prompt_file prompts/prompt_overall_level_with_examples.txt \
    --num_test_sample -1 --temperature 0 --shuffle --use_math_verify --num_workers 32