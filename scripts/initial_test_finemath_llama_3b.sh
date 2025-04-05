python3 math_pipeline/math_eval.py --data_names math,gsm8k --data_dir data \
    --model_name_or_path "HuggingFaceTB/FineMath-Llama-3B" --output_dir initial_inference \
    --prompt_type qwen25-math-cot --split train --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm
