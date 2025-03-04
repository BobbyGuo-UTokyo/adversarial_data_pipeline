import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grader import math_equal_process
from evaluate import evaluate, evaluate_one_file
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
#from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from vlmeval.config import supported_VLM

gsm8k_correct =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208]
math_correct = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 104, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 120, 121, 122, 123, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225]

import os
import json
import random
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import load_jsonl, lower_keys


import os
import json
import random
from anthropic import Anthropic
from typing import Dict, List
from tqdm import tqdm

# Initialize Claude client

assert "ANTHROPIC_API_KEY" in os.environ, "Please set ANTHROPIC_API_KEY environment variable"
anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])




def generate_adversarial_problem(model, question: str, model_answer: str, gt: str) -> Dict:
    """Generate an adversarial math problem using Claude API"""
    
    prompt = f"""As a math teacher, analyze this math question and create a similar but logically equivalent version:

    Original Question: {question}
    Student's Solution: {model_answer}

    Please generate a new question and its solution that:
    1. Tests similar mathematical concepts but uses a different scenario
    2. Maintains mathematical rigor and logical consistency
    3. Involves natural calculations that flow logically 
    4. Uses realistic numbers and scenarios
    5. Shows clear step-by-step reasoning
    6. Arrives at an answer through valid mathematical operations
    7. The final numerical answer should be {gt} through natural mathematical steps
    8. Do not artificially manipulate numbers to force this result - the solution should flow logically
    9. If you cannot create a naturally equivalent problem, notify me that it's not possible to maintain mathematical integrity
    10. Answer should not be rounded or approximated

    Provide your response in this format:
    <new_problem>
    [Your new question here]
    </new_problem>
    explain your reasoning here
    Important: Focus on creating a logically sound problem with clear mathematical steps.
    """

    response = model.generate(prompt)
    
    # Parse response to extract new question
    new_question = response.content[0].text
    
    return {
        "original_question": question,
        "original_answer": model_answer,
        "adversarial_question": new_question
    }
    
def generate_new_solution(model, question: str, model_answer: str, gt: str) -> Dict:
    """Generate solution for the adversarial math problem using Claude API"""
    Final_numerical_answer = "Final numerical answer"
    prompt = f"""As a math teacher, solve this math problem with clear step-by-step reasoning:

    Question: {question}

    Please provide a detailed solution that:
    1. Shows each step clearly
    2. Explains the mathematical reasoning
    3. Uses valid mathematical operations
    4. Arrives at the final answer of {gt}
    5. Uses natural calculations that flow logically
    6. Does not round or approximate numbers
    
    Provide your solution in this format:
    [Your step-by-step solution here] | $\\boxed{Final_numerical_answer}$
    """

    response = model.generate(prompt)
    
    return {
        "question": question,
        "model_answer": model_answer,
        "solution": response.content[0].text
    }

def extract_final_answer(text: str) -> str:
    """Extract numerical answer from text after | symbol"""
    if '|' not in text:
        return ""
    answer_part = text.split('|')[-1].strip()
    if '$\\boxed{' in answer_part:
        # Extract between \boxed{}
        start = answer_part.find('$\\boxed{') + 8
        end = answer_part.find('}', start)
        return answer_part[start:end]
    return answer_part

def extract_new_problem(text: str) -> str:
    """Extract text within <new_problem> tags"""
    start_tag = "<new_problem>"
    end_tag = "</new_problem>"
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)
    return text[start_index:end_index].strip()

def extract_new_problem_solution(text: str) -> str:
    """Extract text within <new_problem_solution> tags"""
    start_tag = "<new_problem_solution>"
    end_tag = "</new_problem_solution>"
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)
    return text[start_index:end_index].strip()

def generate_adversarial_dataset(samples: List[Dict], num_examples: int = 100) -> List[Dict]:
    """Generate a dataset of adversarial problems"""
    
    adversarial_problems = []
    selected_samples = random.sample(samples, num_examples)
    
    for sample in tqdm(selected_samples):
        try:
            adversarial = generate_adversarial_problem(
                sample["question"],
                sample["answer"],
                sample["gt"]
            )
            adversarial_problems.append(adversarial)
        except Exception as e:
            print(f"Error generating adversarial example: {e}")
            continue
            
    return adversarial_problems

def save_adversarial_dataset(problems: List[Dict], output_file: str):
    """Save adversarial problems to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(problems, f, indent=2)


def load_data(data_name, split, data_dir="./data"):
    data_dir = "/share5/ru.wang/code/Qwen2.5-Math/evaluation/outputs_previous/Qwen"
    data_file = f"{data_dir}/Qwen2.5-Math-7B-Instruct/math_eval/{data_name}/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
    print(data_file)
    if os.path.exists(data_file):
        print("load data from file")
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "svamp":
            # evaluate on training set + test set
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets(
                [dataset, load_dataset("ChilleD/SVAMP", split="test")]
            )
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(
                lambda x: ";" not in x["answer"]
            )  # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example["type"] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "mmlu_stem":
            dataset = load_dataset("hails/mmlu_no_train", "all", split="test")
            # only keep stem subjects
            stem_subjects = [
                "abstract_algebra",
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "electrical_engineering",
                "elementary_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_statistics",
                "machine_learning",
            ]
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x["type"] in stem_subjects)
        elif data_name == "carp_en":
            dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output_adversarial", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # Filter examples based on correct indices
    if data_name == "gsm8k":
        examples = [ex for ex in examples if ex["idx"] in gsm8k_correct]
    elif data_name == "math":
        examples = [ex for ex in examples if ex["idx"] in math_correct]

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    for data_name in data_list:
        #new_problems = generate_adversarial_dataset(examples, num_examples=100)
        #main(anthropic, data_name, args)
        main(llm, tokenizer, data_name, args)

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def get_pred(text):
    """Extract numerical answer from text by finding numbers before \boxed or after Final Answer"""
    
    # Handle \boxed{} format with variations
    boxed_patterns = [r'\[\s*\\boxed\{\s*([-+]?\d*\.?\d+(?:\/\d+)?)\s*\}\s*\]', 
                     r'\$\s*\\boxed\{\s*([-+]?\d*\.?\d+(?:\/\d+)?)\s*\}\s*\$',
                     r'\\boxed\{\s*([-+]?\d*\.?\d+(?:\/\d+)?)\s*\}']
    
    for pattern in boxed_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # Look for "Final Answer" followed by number
    match = re.search(r'Final Answer.*?(\d+)', text)
    if match:
        return match.group(1)
        
    # Extract last number in text
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]
        
    return ""


def main(model_name, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print(len(examples), len(gsm8k_correct), len(math_correct))
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])
    # init python executor
    max_func_call = 10
    start_time = time.time()
    result_json = []
    model = supported_VLM[model_name]
    for sample in examples:
        for ind_try in range(max_func_call):
            try:
                response_llm = generate_adversarial_problem(model, tokenizer, sample['question'], sample['answer'], sample['gt'])
                new_problem = extract_new_problem(response_llm['adversarial_question'])
                response_llm_solution = generate_new_solution(model, tokenizer, new_problem, sample['answer'], sample['gt'])
                new_problem_solution = response_llm_solution['solution']
                #new_problem_solution = extract_new_problem_solution(response_llm['adversarial_question'])
                pred = get_pred(new_problem_solution) #extract_final_answer(new_problem_solution)
                print(f"Try {ind_try}: \n{response_llm['adversarial_question']} \nnew problem\n{new_problem} \nnew problem solution: \n{new_problem_solution} \n{pred} == {sample['gt']}?")
                if (math_equal_process((sample['idx'], pred, sample['gt']))):
                    print("Success")
                    result_json.append(
                        {
                            "idx": sample["idx"],
                            "response_llm": response_llm,
                            "problem": sample['question'],
                            "new_problem": new_problem,
                            "new_problem_solution": new_problem_solution,
                            "gt": sample["gt"],
                        }
                    )
                    break
                else:
                    print("Fail")
            except Exception as e:
                print(e)
    
    
    output_file = args.output_dir + f"/adversarial_{data_name}_{args.model_name_or_path.split('/')[-1]}.jsonl"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    print(f"Total examples of adversarial {data_name}: {len(result_json)}")
    print(f"Save to {output_file}")
    with open(output_file, "w") as f:
        for sample in result_json:
            f.write(json.dumps(sample) + "\n")
      
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
