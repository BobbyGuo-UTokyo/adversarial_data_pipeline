import os
import pandas as pd

def process_data(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    assert len(args.adversarial_generation_files) == len(args.test_result_files)
    for ad_f, t_f in zip(args.adversarial_generation_files, args.test_result_files):
        ad_df = pd.read_json(ad_f, orient="records", lines=True).set_index("idx")
        t_df = pd.read_json(t_f, orient="records", lines=True).set_index("idx").drop(columns=["gt"])
        output_df = ad_df.join(t_df, how="left").reset_index().drop_duplicates(subset=["idx"])
        output_file = t_f.replace(".jsonl", "_inspection.xlsx")
        output_df.to_excel(output_file, index=False)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adversarial_generation_files", default="train_qwen25-math-cot_100_seed24678_t0.0_s0_e-1_name_change_only.jsonl", type=str, nargs="+")
    parser.add_argument("--test_result_files", default="train_qwen25-math-cot_100_seed24678_t0.0_s0_e-1_name_change_only_test_result.jsonl", type=str, nargs="+")
    parser.add_argument("--output_dir", default="outputs/data_inspection", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    process_data(args)
