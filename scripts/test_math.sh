python3 math_pipeline/math_eval.py --data_names math,gsm8k --data_dir outputs/adversarial_attack \
    --adversarial_generation_file_name train_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" --output_dir test_adversarial_results \
    --output_name train_qwen25-math-cot_-1_seed0_t0.0_s0_e-1_test_result.jsonl \
    --prompt_type qwen25-math-cot --split train --num_test_sample -1 \
    --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
    --test_adversarial_generation --save_outputs --use_safetensors
# python3 math_pipeline/math_eval.py --data_names math,gsm8k --data_dir output_adversarial \
#     --adversarial_generation_file_name train_qwen25-math-cot_100_seed24678_t0.0_s0_e-1_paraphrase.jsonl \
#     --model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" --output_dir test_adversarial_results \
#     --output_name train_qwen25-math-cot_100_seed24678_t0.0_s0_e-1_paraphrase_test_result.jsonl \
#     --prompt_type qwen25-math-cot --split train --num_test_sample 100 --seed 24678 \
#     --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
#     --test_adversarial_generation --save_outputs --use_safetensors
# python3 math_pipeline/math_eval.py --data_names math,gsm8k --data_dir output_adversarial \
#     --adversarial_generation_file_name train_qwen25-math-cot_100_seed24678_t0.0_s0_e-1_full_change.jsonl \
#     --model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" --output_dir test_adversarial_results \
#     --output_name train_qwen25-math-cot_100_seed24678_t0.0_s0_e-1_full_change_test_result.jsonl \
#     --prompt_type qwen25-math-cot --split train --num_test_sample 100 --seed 24678 \
#     --start 0 --end -1 --temperature 0 --shuffle --use_math_verify \
#     --test_adversarial_generation --save_outputs --use_safetensors
