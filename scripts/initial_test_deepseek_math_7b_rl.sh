python3 math_pipeline/math_eval.py --data_names math,gsm8k --data_dir data \
    --model_name_or_path "deepseek-ai/deepseek-math-7b-rl" --output_dir initial_inference \
    --prompt_type mathstral --split train --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm
