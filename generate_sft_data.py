import json
import os
import pandas as pd

def generate_sft_data(args):
    df = pd.read_excel(args.input_file)
    potential_outputs = {
        "original_pos": df[df.apply(lambda x: json.loads(x["score"].lower())[0], axis=1)].rename(columns={"original_question": "sft_prompt", "original_solution": "sft_response"}),
        "original_neg": df[df.apply(lambda x: not json.loads(x["score"].lower())[0], axis=1)].rename(columns={"original_question": "sft_prompt", "original_solution": "sft_response"}),
        "adversarial_pos": df[df.apply(lambda x: json.loads(x["score"].lower())[0], axis=1)].rename(columns={"question": "sft_prompt", "gt_cot": "sft_response"}),
        "adversarial_neg": df[df.apply(lambda x: not json.loads(x["score"].lower())[0], axis=1)].rename(columns={"question": "sft_prompt", "gt_cot": "sft_response"})
    }
    output_df = pd.concat([potential_outputs[sample_type] for sample_type in args.included_sample_types])
    output_df = output_df[["idx", "sft_prompt", "sft_response"]]
    if args.continue_pretrain:
        output_df["sft_continue_pretrain_text"] = output_df["sft_prompt"] + "\n\n" + output_df["sft_response"]
        output_df = output_df[["idx", "sft_continue_pretrain_text"]]
    output_dir = os.path.dirname(args.output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) # create output directory if not exists
    # compose output file name
    output_name = os.path.basename(args.output_file).split(".")[0]
    for sample_type in args.included_sample_types:
        output_name += f"_{sample_type}"
    if args.continue_pretrain:
        output_name += "_continue_pretrain"
    output_name += "." + os.path.basename(args.output_file).split(".")[1]
    output_file = os.path.join(output_dir, output_name)
    output_df.to_json(output_file, orient="records", lines=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--included_sample_types", type=str, nargs="+", required=True)
    parser.add_argument("--continue_pretrain", action="store_true")
    args = parser.parse_args()
    
    generate_sft_data(args)
