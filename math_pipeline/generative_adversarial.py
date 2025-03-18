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
import sys
# Add root directory to system path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from vlmeval.config import supported_VLM
from math_verify import parse, verify

gsm8k_correct =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208]
math_correct = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 104, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 120, 121, 122, 123, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225]
# gsm8k_correct = []
# math_correct = []

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


def generate_adversarial_problem(model, question: str, model_answer: str, granularity_prompt: str, gt: str) -> Dict:
    """Generate an adversarial math problem using API"""
    
    prompt = f"""As a math teacher, analyze this math question and create a similar but logically equivalent version:

    Original Question: {question}
    Student's Solution: {model_answer}

    Please generate a new question and its solution that:
    1. Tests similar mathematical concepts
    2. {granularity_prompt}
    3. Maintains mathematical rigor and logical consistency
    4. Involves natural calculations that flow logically 
    5. Uses realistic numbers and scenarios
    6. Shows clear step-by-step reasoning
    7. Arrives at an answer through valid mathematical operations
    8. The final numerical answer should be {gt} through natural mathematical steps
    9. Do not artificially manipulate numbers to force this result - the solution should flow logically
    10. If you cannot create a naturally equivalent problem, notify me that it's not possible to maintain mathematical integrity
    11. Answer should not be rounded or approximated

    Provide your response in this format:
    <new_problem>
    [Your new question here. Question only. No solution.]
    </new_problem>
    explain your reasoning here
    Important: Focus on creating a logically sound problem with clear mathematical steps.
    """

    response, reasoning = model.generate(prompt)
    
    # # Parse response to extract new question
    # new_question = response.content[0].text
    
    return {
        "original_question": question,
        "original_answer": model_answer,
        "adversarial_question": response,
        "adversarial_reasoning": reasoning
    }

def generate_adversarial_problem_given_prompt_file(model, question: str, model_answer: str, gt: str, prompt_file: str) -> Dict:
    """Generate an adversarial math problem using API"""
    
    lines = open(prompt_file).readlines()
    prompt = "\n".join(open(prompt_file).readlines()).format(question, gt, model_answer, gt)
    response, reasoning = model.generate(prompt)

    return {
        "original_question": question,
        "original_answer": model_answer,
        "adversarial_question": response,
        "adversarial_reasoning": reasoning
    }

def generate_adversarial_problem_word_level(model, question: str, model_answer: str, gt: str) -> Dict:
    prompt = f"""
    Adversarial Math Problem Generator
    Objective: Modify the original problem at the word level to create a semantically equivalent but linguistically distinct version that exposes memorization-based problem-solving and tests deep conceptual understanding.

    Original Question: {question}
    Original ground truth: {gt}
    Student's Solution: {model_answer}

    Steps to Generate Adversarial Problem:

    1. Deconstruct the Original Problem:
    - Identify the core mathematical concept (e.g., linear equations, geometry).
    - Map the problem's logical structure (e.g., sequential steps, dependencies).
    - Identify potential pitfalls or shortcuts that a solver might exploit if they focus only on surface-level cues.

    2. Analyze the student's solution:
    - Check the validity of the solution and verify that the reasoning truly understands the core mathematical concepts and steps are naturally leading to the final answer.
    - If the answer is correct, find potential shortcuts or assumptions the student used (e.g., memorized formulas without contextual analysis).
    - If the answer is incorrect, diagnose the root misunderstanding (e.g., misapplying order of operations, misinterpreting units).

    3. Word-Level Adversarial Modifications:
    - Disassemble the original problem word by word.
    - Replace or tweak each word to integrate adversarial elements that can create fine-grained confusion or introduce subtle traps.
    - Ensure that while the individual words change, the underlying computational logic remains unchanged.

    4. Rebuild Context & Structure:
    - Transplant the problem into a new real-world scenario (e.g., replace "bakery sales" with "bookstore inventory").
    - Retain the original problem format (e.g., if the original uses a two-part question, mirror it exactly).
    - Ensure that the new context still embeds the identified pitfalls in specific words.

    5. Embed Pitfalls Strategically:
    - Choose words that may have multiple interpretations or ambiguous meanings to prompt common mistakes or encourage reliance on shortcuts.
    - Consider embedding potential edge cases or misleading hints by carefully selecting synonyms that challenge solvers.

    6. Validation Check:
    - Confirm the adversarial problem:
        -- Requires the same mathematical logic as the original.
        -- Contains no contradictions or mismatched values.
        -- Would trick memorization-based solvers but reward conceptual understanding.
        -- The final answer remains {gt} through valid steps, even with modified wording.
    - If validation fails, return: Error: Adversarial problem cannot satisfy constraints.

    Output Format:

    Your final output should be wrapped in the following tags:

    <new_problem>
    Insert your generated problem description here.
    [Insert adversarially modified problem here. Maintain the original’s structure, including line breaks, punctuation, and numbering.]
    </new_problem>

    Example:
    Original Problem:
    Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?

    Adversarial Problem:
    <new_problem>
    David is rendering a 200 GB video project. Normally his software can render 2 GB/minute, but 40% of the way through the process, the software crashes, requiring a system reboot that takes 20 minutes. After rebooting, David has to restart the rendering from the beginning. How long does it take to complete the rendering?
    </new_problem>
    """
    response, reasoning = model.generate(prompt)
    
    return {
        "original_question": question,
        "original_answer": model_answer,
        "adversarial_question": response,
        "adversarial_reasoning": reasoning
    }

def generate_new_solution(model, question: str, has_reasoning: bool=True) -> Dict:
    """Generate solution for the adversarial math problem using Claude API"""
    Final_numerical_answer = "Final numerical answer"
    prompt = f"""As a math teacher, solve this math problem with clear step-by-step reasoning:

    Question: {question}

    Please provide a detailed solution that:
    1. Shows each step clearly
    2. Explains the mathematical reasoning
    3. Uses valid mathematical operations
    4. Uses natural calculations that flow logically
    5. Does not round or approximate numbers
    
    Provide your solution in this format:
    [Your step-by-step solution here] | $\\boxed{Final_numerical_answer}$
    """

    if has_reasoning:
        response, reasoning = model.generate(prompt)
    else:
        response = model.generate(prompt)
        reasoning = ""
    
    return {
        "question": question,
        "solution": response,
        "reasoning": reasoning
    }

def generate_new_solution_given_gt(model, question: str, gt: str, has_reasoning: bool=True) -> Dict:
    """Generate solution for the adversarial math problem using Claude API"""
    Final_numerical_answer = "Final numerical answer"
    prompt = f"""As a math teacher, solve this math problem with clear step-by-step reasoning:

    Question: {question}

    Please provide a detailed solution that:
    1. Shows each step clearly
    2. Explains the mathematical reasoning
    3. Uses valid mathematical operations
    4. Uses natural calculations that flow logically
    5. Does not round or approximate numbers
    6. Arrives at the final answer of {gt}
    7. Do not artificially manipulate numbers or add conditions to force this result - the solution should flow logically
    
    Provide your solution in this format:
    [Your step-by-step solution here] | $\\boxed{Final_numerical_answer}$
    """

    if has_reasoning:
        response, reasoning = model.generate(prompt)
    else:
        response = model.generate(prompt)
        reasoning = ""
    
    return {
        "question": question,
        "solution": response,
        "reasoning": reasoning
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
    data_dir = "../ru.wang/code/Qwen2.5-Math/evaluation/outputs_previous/Qwen"
    data_file = f"{data_dir}/Qwen2.5-Math-7B-Instruct/math_eval/{data_name}/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
    # data_file = f"{data_dir}/{data_name}/{split}.jsonl"
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
    # add 'question'
    if "question" not in examples[0]:
        examples = [
            {**example, "question": parse_question(example, data_name)}
            for example in examples
        ]
    # add 'gt'
    if "gt" not in examples[0]:
        examples = [
            {**example, "gt": parse_ground_truth(example, data_name)[1], "answer": parse_ground_truth(example, data_name)[0]}
            for example in examples
        ]
    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="VolcEngine_DeepSeekR1", type=str)
    parser.add_argument("--verifier_model_names", default=["VolcEngine_DeepSeekR1"], type=str, nargs="+")
    parser.add_argument("--endpoint_id", default='ep-20250216235228-69vhs', type=str)
    parser.add_argument("--output_dir", default="./output_adversarial", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--granularity_prompt", default="Mainly modify variable names like item names or person names and try to keep other things unchanged", type=str)
    parser.add_argument("--prompt_file", default=None, type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--postfix", default="", type=str)
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
    parser.add_argument("--use_math_verify", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--max_func_call", type=int, default=3)
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
    if data_name == "gsm8k" and len(gsm8k_correct) > 0:
        examples = [ex for ex in examples if ex["idx"] in gsm8k_correct]
    elif data_name == "math" and len(math_correct) > 0:
        examples = [ex for ex in examples if ex["idx"] in math_correct]

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        # random.seed(datetime.now().timestamp())
        random.seed(args.seed)
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
    if len(args.postfix):
        post_fix = f"_{args.postfix}"
    else:
        post_fix = ""
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}{post_fix}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite and os.path.isfile(out_file):
        # processed_files = [
        #     f
        #     for f in os.listdir(f"{output_dir}/{data_name}/")
        #     if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        # ]
        # for f in processed_files:
        #     processed_samples.extend(
        #         list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
        #     )
        processed_samples = list(load_jsonl(out_file))

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # # load model
    # available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    # if args.use_vllm:
    #     llm = LLM(
    #         model=args.model_name_or_path,
    #         tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
    #         pipeline_parallel_size=args.pipeline_parallel_size,
    #         trust_remote_code=True,
    #     )
    #     tokenizer = None
    #     if args.apply_chat_template:
    #         tokenizer = AutoTokenizer.from_pretrained(
    #             args.model_name_or_path, trust_remote_code=True
    #         )
    # else:
    #     llm, tokenizer = load_hf_lm_and_tokenizer(
    #         model_name_or_path=args.model_name_or_path,
    #         load_in_half=True,
    #         use_fast_tokenizer=True,
    #         use_safetensors=args.use_safetensors,
    #     )

    # infer & eval
    data_list = args.data_names.split(",")
    for data_name in data_list:
        #new_problems = generate_adversarial_dataset(examples, num_examples=100)
        #main(anthropic, data_name, args)
        # main(llm, tokenizer, data_name, args)
        main(args.model_name_or_path, None, data_name, args)

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
    max_func_call = args.max_func_call
    start_time = time.time()
    # result_json = []
    result_dict = {}

    # Initialize DeepSeek model
    if model_name.startswith("VolcEngine"):
        assert args.endpoint_id is not None and isinstance(args.endpoint_id, str)
        model = supported_VLM[model_name](model=args.endpoint_id, has_reasoning=True, temperature=0, retry=3, verbose=False)
    # Prepare output
    # output_file = args.output_dir + f"/math/adversarial_{data_name}_{args.model_name_or_path.split('/')[-1]}_{args.split}.jsonl"

    # Initialize verifier models
    verifier_models = []
    for verifier_model_name in args.verifier_model_names:
        if verifier_model_name.startswith("VolcEngine"):
            assert args.endpoint_id is not None and isinstance(args.endpoint_id, str)
            verifier_model = supported_VLM[verifier_model_name](model=args.endpoint_id, has_reasoning=True, temperature=0, retry=3, verbose=False)
            verifier_models.append(verifier_model)
        else:
            verifier_model = supported_VLM[verifier_model_name]()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # Iterate through examples
    with open(out_file, "w" if args.overwrite else "a") as f:
        for sample in examples:
            sample_succeeded = False
            for ind_try in range(max_func_call):
                try:
                    # response_llm = generate_adversarial_problem(model, tokenizer, sample['question'], sample['answer'], sample['gt'])
                    # response_llm = generate_adversarial_problem(model, sample['question'], sample['answer'], args.granularity_prompt, sample['gt'])
                    # response_llm = generate_adversarial_problem_word_level(model, sample['question'], sample['answer'], sample['gt'])
                    if args.prompt_file is not None:
                        response_llm = generate_adversarial_problem_given_prompt_file(model, sample['question'], sample['answer'], sample['gt'], args.prompt_file)
                    else:
                        response_llm = generate_adversarial_problem(model, sample['question'], sample['answer'], args.granularity_prompt, sample['gt'])
                    new_problem = extract_new_problem(response_llm['adversarial_question'])
                    adversarial_reasoning = response_llm['adversarial_reasoning']
                    # response_llm_solution = generate_new_solution(model, tokenizer, new_problem, sample['answer'], sample['gt'])
                    # response_llm_solution = generate_new_solution(model, new_problem, sample['answer'], sample['gt'])
                    response_llm_solution = generate_new_solution(model, new_problem, has_reasoning=True)
                    new_problem_solution = response_llm_solution['solution']
                    new_problem_reasoning = response_llm_solution['reasoning']
                    #new_problem_solution = extract_new_problem_solution(response_llm['adversarial_question'])
                    # # Verify the new problem solution with verifier models
                    # verifier_solutions = []
                    # for verifier_model in verifier_models:
                    #     verifier_response = generate_new_solution(verifier_model, new_problem)
                    #     verifier_pred = get_pred(verifier_response['solution'])
                    #     if not verify(parse(sample["gt"]), parse(verifier_pred)):
                    #         print(f"Verifier {verifier_model.__class__.__name__} failed to verify the new problem solution.")
                    #         break
                    # verified_correct = True
                    if args.use_math_verify:
                        pred = parse(new_problem_solution)
                        verified_correct = verify(parse(sample["gt"]), pred)
                        print(f"Try {ind_try}: \noriginal_question: {sample['question']} \n{response_llm['adversarial_question']} \nnew problem\n{new_problem} \nnew problem solution: \n{new_problem_solution} \n{pred} == {parse(sample['gt'])}?")
                    else:
                        pred = get_pred(new_problem_solution) #extract_final_answer(new_problem_solution)
                        print(f"Try {ind_try}: \noriginal_question: {sample['question']} \n{response_llm['adversarial_question']} \nnew problem\n{new_problem} \nnew problem solution: \n{new_problem_solution} \n{pred} == {sample['gt']}?")
                        verified_correct = math_equal_process((sample['idx'], pred, sample['gt']))
                    if verified_correct:
                        sample_succeeded = True
                        print("Success")
                        success_result = {
                            "idx": sample["idx"],
                            "response_llm": response_llm,
                            "problem": sample['question'],
                            "adversarial_reasoning": adversarial_reasoning,
                            "new_problem": new_problem,
                            "new_problem_solution": new_problem_solution,
                            "new_problem_reasoning": new_problem_reasoning,
                            "gt": sample["gt"],
                        }
                        result_dict[sample["idx"]] = success_result
                        break
                    else:
                        print("Fail")
                except Exception as e:
                    print(e)
            if sample_succeeded:
                f.write(json.dumps(success_result) + "\n")
    print(f"Total examples of adversarial {data_name}: {len(result_dict)}")
    print(f"Save to {out_file}")
      
    return list(result_dict.values())


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
