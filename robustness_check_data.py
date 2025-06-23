import json
import os
import pandas as pd

def generate_robustness_check_data(args):
    df = pd.read_excel(args.input_file)
    potential_outputs = {
        "original_pos": df[df.apply(lambda x: json.loads(x["score"].lower())[0], axis=1)].drop(columns=["gt"]).rename(columns={"original_question": "new_problem", "original_gt_cot": "new_problem_solution", "original_answer": "gt"})[["idx", "new_problem", "new_problem_solution", "gt"]],
        "original_neg": df[df.apply(lambda x: not json.loads(x["score"].lower())[0], axis=1)].drop(columns=["gt"]).rename(columns={"original_question": "new_problem", "original_gt_cot": "new_problem_solution", "original_answer": "gt"})[["idx", "new_problem", "new_problem_solution", "gt"]],
        "adversarial_pos": df[df.apply(lambda x: json.loads(x["score"].lower())[0], axis=1)].rename(columns={"question": "new_problem", "gt_cot": "new_problem_solution"})[["idx", "new_problem", "new_problem_solution", "gt"]],
        "adversarial_neg": df[df.apply(lambda x: not json.loads(x["score"].lower())[0], axis=1)].rename(columns={"question": "new_problem", "gt_cot": "new_problem_solution"})[["idx", "new_problem", "new_problem_solution", "gt"]],
    }
    print(f"Loaded {len(df)} samples from {args.input_file}")
    output_dir = os.path.dirname(args.output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) # create output directory if not exists
    selected_outputs = potential_outputs[args.indices_sample_type]
    selected_indices = json.load(open(args.indices_file))["sampled_test"]
    selected_outputs = selected_outputs[selected_outputs["idx"].isin(selected_indices)]
    print(f"Loaded {len(selected_outputs)} {args.indices_sample_type} samples from {args.input_file}")
    output_name = os.path.basename(args.output_file).split(".")[0] + f"_{args.indices_sample_type}.jsonl"
    output_file = os.path.join(output_dir, output_name)
    selected_outputs.to_json(output_file, orient="records", lines=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--indices_sample_type", type=str, required=True)
    parser.add_argument("--indices_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    generate_robustness_check_data(args)
