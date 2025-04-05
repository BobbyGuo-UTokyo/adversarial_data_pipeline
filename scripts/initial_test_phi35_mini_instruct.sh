python3 math_pipeline/math_eval.py --data_names math,gsm8k --data_dir data \
    --model_name_or_path "microsoft/Phi-3.5-mini-instruct" --output_dir initial_inference \
    --prompt_type mathstral --split train --num_test_sample 10 --seed 24678 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --save_outputs --use_safetensors --use_vllm
