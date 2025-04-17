import os
import pandas as pd

def process_data(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    assert len(args.adversarial_test_result_files) == len(args.original_test_result_files)
    for adversarial_f, original_f in zip(args.adversarial_test_result_files, args.original_test_result_files):
        ad_df = pd.read_json(adversarial_f, orient="records", lines=True).set_index("idx")
        t_df = pd.read_json(original_f, orient="records", lines=True).set_index("idx").drop(columns=["gt"])

        t_df = t_df.rename(columns={"question": "original_question", "answer": "original_answer", "solution": "original_solution", "gt_cot": "original_gt_cot", "prompt": "original_prompt", "code": "original_code", "pred": "original_pred", "report": "original_report", "parsed_pred": "original_parsed_pred", "parsed_gt": "original_parsed_gt", "score": "original_score"})
        if "is_adversarial_generation" in t_df.columns:
            t_df = t_df.drop(columns=["is_adversarial_generation"])
        
        output_df = ad_df.join(t_df, how="left").reset_index().drop_duplicates(subset=["idx"])
        output_file = adversarial_f.replace(".jsonl", "_inspection.xlsx")
        output_df.to_excel(output_file, index=False)
        print(output_file)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adversarial_test_result_files", default="train_qwen25-math-cot_100_seed24678_t0.0_s0_e-1_name_change_only.jsonl", type=str, nargs="+")
    parser.add_argument("--original_test_result_files", default="train_qwen25-math-cot_100_seed24678_t0.0_s0_e-1_name_change_only_test_result.jsonl", type=str, nargs="+")
    parser.add_argument("--output_dir", default="outputs/data_inspection", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    process_data(args)
