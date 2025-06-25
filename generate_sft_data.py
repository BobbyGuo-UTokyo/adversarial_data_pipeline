import json
import os
import pandas as pd

def generate_sft_data(args):
    if args.input_file.endswith(".xlsx"):
        df = pd.read_excel(args.input_file)
    elif args.input_file.endswith(".jsonl"):
        df = pd.read_json(args.input_file, orient="records", lines=True)
    answer_str = " The answer is $\\boxed{{{}}}"
    if args.input_file_type == "adversarial":
        if "original_solution" not in df.columns: # for gsm8k
            df["original_solution"] = df.apply(lambda x: x["original_answer"].split("####")[0] + answer_str.format(x["original_answer"].split("####")[1].strip()), axis=1)
        potential_outputs = {
            "original_pos": df[df.apply(lambda x: json.loads(x["score"].lower())[0], axis=1)].rename(columns={"original_question": "sft_prompt", "original_solution": "sft_response"}),
            "original_neg": df[df.apply(lambda x: not json.loads(x["score"].lower())[0], axis=1)].rename(columns={"original_question": "sft_prompt", "original_solution": "sft_response"}),
            "adversarial_pos": df[df.apply(lambda x: json.loads(x["score"].lower())[0], axis=1)].rename(columns={"question": "sft_prompt", "gt_cot": "sft_response"}),
            "adversarial_neg": df[df.apply(lambda x: not json.loads(x["score"].lower())[0], axis=1)].rename(columns={"question": "sft_prompt", "gt_cot": "sft_response"})
        }
    elif args.input_file_type == "original_gsm8k":
        df["answer"] = df.apply(lambda x: x["answer"].split("####")[0] + answer_str.format(x["answer"].split("####")[1].strip()), axis=1)
        potential_outputs = {
            "original_full": df.rename(columns={"question": "sft_prompt", "answer": "sft_response"})
        }
    elif args.input_file_type == "original_math":
        if "idx" not in df.columns:
            df["idx"] = df.index + 1
        potential_outputs = {
            "original_full": df.rename(columns={"problem": "sft_prompt", "solution": "sft_response"})
        }
    print(f"Loaded {len(df)} samples from {args.input_file}")
    # load neg from extra input file
    if len(args.extra_input_files) > 0 and len(args.extra_input_sample_types) > 0:
        assert len(args.extra_input_sample_types) == len(args.extra_input_files)
        for extra_input_file, extra_input_sample_type in zip(args.extra_input_files, args.extra_input_sample_types):
            assert extra_input_sample_type not in potential_outputs
            if extra_input_file.endswith(".jsonl"):
                extra_df = pd.read_json(extra_input_file, orient="records", lines=True)
            elif extra_input_file.endswith(".xlsx"):
                extra_df = pd.read_excel(extra_input_file)
            potential_outputs[extra_input_sample_type] = extra_df[extra_df.apply(lambda x: json.loads(x["score"].lower())[0] if "pos" in extra_input_sample_type else not json.loads(x["score"].lower())[0],
                                                                axis=1)].rename(columns={"question": "sft_prompt", "gt_cot": "sft_response"})
    all_sample_types = args.included_sample_types + args.extra_input_sample_types
    # split data for robustness check. before resample_neg
    robustness_sample_indices = {"sampled_train": [], "sampled_test": []}
    if args.split_robustness > 0:
        for sample_type in all_sample_types:
            if "adversarial_neg" in sample_type:
                original_sample_num = len(potential_outputs[sample_type])
                robustness_sampled_train = potential_outputs[sample_type].sample(frac=args.split_robustness, replace=False, random_state=42)
                robustness_sample_indices["sampled_train"].extend(robustness_sampled_train["idx"].tolist())
                # record unsampled as test
                robustness_sample_indices["sampled_test"].extend(potential_outputs[sample_type]["idx"][~potential_outputs[sample_type]["idx"].isin(robustness_sampled_train["idx"])].tolist())
                potential_outputs[sample_type] = robustness_sampled_train
                print(f"Splitted {len(potential_outputs[sample_type])} adversarial_neg samples from {original_sample_num} {sample_type} data for rubustness evaluation as: {len(potential_outputs[sample_type])} train and {len(robustness_sample_indices['sampled_test'])} test")
    # resample negative samples by the given ratio
    if args.resample_neg > 0:
        for sample_type in all_sample_types:
            if "neg" in sample_type:
                original_sample_num = len(potential_outputs[sample_type])
                potential_outputs[sample_type] = potential_outputs[sample_type].sample(frac=args.resample_neg, replace=True if args.resample_neg > 1.0 else False, random_state=42)
                print(f"Resampled {len(potential_outputs[sample_type])} negative samples from {original_sample_num} {sample_type} data")
    output_df = pd.concat([potential_outputs[sample_type] for sample_type in all_sample_types])
    for sample_type in all_sample_types:
        print(f"Using {len(potential_outputs[sample_type])} {sample_type} samples")
    if args.reindex:
        output_df["idx"] = output_df.index + 1
    output_df = output_df[["idx", "sft_prompt", "sft_response"]]
    if args.continue_pretrain:
        output_df["sft_continue_pretrain_text"] = output_df["sft_prompt"] + "\n\n" + output_df["sft_response"]
        output_df = output_df[["idx", "sft_continue_pretrain_text"]]
    output_dir = os.path.dirname(args.output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) # create output directory if not exists
    # compose output file name
    output_name = os.path.basename(args.output_file).split(".")[0]
    for sample_type in all_sample_types:
        output_name += f"_{sample_type}"
    for sample_type in args.extra_input_sample_types:
        if sample_type not in all_sample_types:
            output_name += f"_{sample_type}"
    if args.continue_pretrain:
        output_name += "_continue_pretrain"
    if args.resample_neg > 0:
        output_name += f"_resample_neg_{args.resample_neg}".replace(".", "_")
    if args.split_robustness > 0:
        output_name += f"_split_robustness_{args.split_robustness}".replace(".", "_")
        with open(os.path.join(output_dir, output_name + "_sample_indices.json"), "w") as f:
            json.dump(robustness_sample_indices, f, indent="\t")
    output_name += "." + os.path.basename(args.output_file).split(".")[1]
    output_file = os.path.join(output_dir, output_name)
    output_df.to_json(output_file, orient="records", lines=True)
    print(f"Saved {len(output_df)} samples to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--input_file_type", type=str, default="adversarial", choices=["original_math", "original_gsm8k", "adversarial"])
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--included_sample_types", type=str, nargs="+", required=True)
    parser.add_argument("--extra_input_files", type=str, nargs="+", default=[])
    parser.add_argument("--extra_input_sample_types", type=str, nargs="+", default=[])
    parser.add_argument("--continue_pretrain", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--split_robustness", type=float, default=-1.0, help="split robustness, negative means no split")
    parser.add_argument("--resample_neg", type=float, default=-1, help="resample negative samples to the given ratio, negative means no resample")
    args = parser.parse_args()
    
    generate_sft_data(args)
