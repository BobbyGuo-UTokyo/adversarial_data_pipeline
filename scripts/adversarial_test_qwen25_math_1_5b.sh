python3 math_pipeline/generative_adversarial.py --data_names math,gsm8k --data_dir outputs/initial_inference \
    --attacker_model_name_or_path DeepSeekR1 \
    --original_model_name_or_path "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --output_dir outputs/adversarial_attack --prompt_type qwen25-math-cot --split train \
    --prompt_file prompts/prompt_overall_level_with_examples.txt --answer_key code \
    --num_test_sample -1 --temperature 0 --shuffle --use_math_verify --num_workers 32