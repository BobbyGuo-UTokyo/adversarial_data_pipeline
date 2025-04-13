python3 math_pipeline/math_eval.py --data_names math --data_dir data \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" \
    --output_dir test_adversarial_results/math_0411/original_no_train \
    --prompt_type qwen25-math-cot --split test --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm
