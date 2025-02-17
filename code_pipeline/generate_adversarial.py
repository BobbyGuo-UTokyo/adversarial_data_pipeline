import os
import argparse
from pathlib import Path
from typing import Union, Dict
from anthropic import Anthropic
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from evalplus import evaluate
import json

HumanEval_Correct = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 128, 131, 133, 135, 136, 137, 138, 139, 140, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162]
Mbpp_Correct = [2, 3, 4, 6, 7, 8, 11, 12, 14, 16, 17, 18, 19, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 70, 71, 75, 77, 79, 80, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 104, 105, 106, 108, 111, 113, 116, 118, 119, 120, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 139, 140, 142, 145, 160, 161, 162, 165, 166, 167, 168, 170, 171, 172, 222, 223, 224, 226, 227, 230, 232, 233, 234, 238, 239, 240, 242, 245, 247, 250, 251, 252, 253, 255, 256, 257, 259, 261, 262, 264, 265, 266, 267, 269, 270, 271, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 290, 292, 293, 294, 296, 297, 299, 308, 309, 312, 388, 389, 390, 391, 392, 394, 395, 397, 404, 405, 406, 409, 410, 412, 413, 414, 418, 419, 420, 421, 422, 424, 425, 426, 427, 428, 432, 435, 436, 439, 440, 441, 445, 446, 447, 451, 453, 454, 455, 456, 457, 458, 459, 460, 463, 465, 470, 471, 472, 473, 474, 475, 476, 478, 479, 554, 555, 556, 557, 558, 559, 560, 562, 563, 564, 565, 566, 567, 568, 569, 573, 577, 578, 579, 583, 585, 586, 587, 588, 589, 591, 592, 593, 594, 596, 598, 599, 600, 602, 604, 605, 606, 607, 608, 611, 612, 614, 616, 618, 619, 620, 623, 624, 628, 629, 630, 632, 633, 635, 638, 639, 641, 643, 644, 720, 721, 722, 723, 724, 725, 726, 728, 730, 731, 732, 733, 734, 735, 736, 740, 741, 742, 743, 744, 745, 748, 749, 750, 751, 752, 753, 754, 757, 758, 759, 760, 762, 763, 764, 766, 767, 770, 771, 772, 773, 775, 778, 781, 782, 784, 785, 786, 787, 788, 790, 791, 792, 793, 796, 797, 798, 799, 800, 803, 804, 805, 806, 807, 808, 809]
#HumanEval_Correct = [0]
#Mbpp_Correct = [2]

def load_model_outputs(workdir: str, problem_id: str) -> list:
    """Load all model outputs for a given problem"""
    problem_dir = os.path.join(workdir, problem_id.replace("/", "_"))
    outputs = []
    print("problem_dir: ", problem_dir)
    for f in os.listdir(problem_dir):
        if f.endswith(".py"):
            with open(os.path.join(problem_dir, f), "r", encoding="utf-8") as fp:
                outputs.append(fp.read())
    return outputs

def extract_problem_description(text: str) -> str:
    """Extract problem description between [new_problem] tags"""
    start = text.find("<new_problem>\n```python\n")
    if start == -1:
        return ""
    start += len("<new_problem>\n```python\n")
    
    end = text.find("```", start)
    if end == -1:
        return ""
        
    return text[start:end].rstrip()

def extract_solution(text: str) -> str:
    """Extract solution code between [new_problem_solution] tags"""
    start = text.find("<new_problem_solution>\n```python\n")
    if start == -1:
        return ""
    start += len("<new_problem_solution>\n```python\n")
    
    end = text.find("```", start)
    if end == -1:
        return ""
        
    return text[start:end].rstrip()

def evaluate_solution(problem_id: str, solution: str, dataset: str = "humaneval") -> bool:
    """Evaluate if solution passes all test cases"""
    # Create temporary file with solution
    tmp_dir = "tmp_solutions"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{problem_id.replace('/', '_')}.py")
    print("tmp_file: ", tmp_file)
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(solution)
    try:
        # Run evaluation
        print("problem_id: ", problem_id)
        results = evaluate.evaluate_one_file(dataset=dataset, samples_file=tmp_file, task_id=problem_id, base_only=False, i_just_wanna_run=True)
        
        # Check results

            
        # Clean up
        #os.remove(tmp_file)
        #os.remove(result_file)
        
        # Check if passed all tests
        # if dataset == "humaneval":
        #     problem_id = "HumanEval/" + problem_id
        # elif dataset == "mbpp":
        #     problem_id = "mbpp/" + problem_id
        task_results = results["eval"][problem_id][0]
        print("task_results: ", task_results)
        return task_results["base_status"] == "pass" and task_results["plus_status"] == "pass"
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return False
    
def answer_question_via_claude(question: str, claude_client: Anthropic) -> str:
    """Get an answer to a question using Claude API"""
    
    context = f'''You are an expert programmer. I will show you a programming problem below:

    {question}

    Please analyze this problem and provide a correct solution. Your implementation should:

    1. Be robust and handle all edge cases
    2. Have clear and descriptive comments
    3. Import any necessary libraries
    4. Focus on correctness rather than optimization

    Present your solution in this exact format:

    <new_problem_solution>
    ```python
    # Your complete implementation here, including imports
    ```
    </new_problem_solution>
'''
    
    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=1,
        messages=[{"role": "user", "content": context}]
    )
    return response.content[0].text

def generate_adversarial_prompt(original_prompt: str, model_outputs: list, claude_client: Anthropic) -> str:
    """Generate adversarial prompt using Claude API"""
    context = f'''You are an expert computer science professor who specializes in identifying students' misconceptions and creating targeted practice problems. I will provide you with:

    An original programming problem {original_prompt}
    A solution attempt from an LLM {model_outputs}

    Your task is to generate an adversarial programming problem that reveals potential flaws or shortcuts in the LLM's solution approach while maintaining the core mathematical/computational requirements of the original problem.

    SOLUTION ANALYSIS
    First, analyze the provided LLM solution for:
    - Potential algorithmic shortcuts or assumptions
    - Edge cases that might be missed 
    - Implementation patterns suggesting misunderstanding
    - Over-reliance on specific data structures/algorithms
    - Special case handling

    IDENTIFY EXPLOITATION OPPORTUNITIES
    Based on your analysis, identify:
    - Which assumptions could be challenged
    - What edge cases could reveal flaws
    - How the solution might fail while passing original tests
    - What problem-solving patterns are overly relied upon

    GENERATE NEW PROBLEM
    Create a new problem that:
    - MUST maintain exact same functionality - same input MUST produce same output as original problem
    - Uses different domain terminology/context
    - Keep the original test cases for validation
    - Frames problem to discourage problematic shortcuts

    OUTPUT FORMAT
    Present your response in this format:

    <new_problem>
    ```python
    (Insert your generated problem description including:
    - Function signature matching original I/O types and behavior
    - Problem description 
    - Keep original test cases)
    ```
    </new_problem>

    EXPLANATION:
    - List key flaws identified in original solution
    - Describe all edge cases and special cases that should be handled
    - Explain how new problem exposes these flaws
    - Describe how terminology changes discourage shortcuts
    - Show mathematical equivalence to original problem

    Remember:
    - Same inputs MUST produce identical outputs as original function
    - Keep exact same input/output types as original
    - Keep original test cases
    - Focus on exposing flaws while maintaining mathematical equivalence'''

    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=1,
        messages=[{"role": "user", "content": context}]
    )
    
    return response.content[0].text

def adversarial_generate(args):
    claude = Anthropic(api_key=args.claude_key)
    
    with Progress(
        TextColumn(f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        id_range = None
        if args.dataset == "humaneval":
            from evalplus.data import get_human_eval_plus
            dataset = get_human_eval_plus()
            id_range = HumanEval_Correct
        elif args.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus
            dataset = get_mbpp_plus()
            id_range = Mbpp_Correct
        os.makedirs(args.output_dir, exist_ok=True)
        
        for task_id, task in p.track(dataset.items()):
            #if args.id_range:
            id_num = int(task_id.split("/")[1])
            #low, high = args.id_range
            # if id_num < low or id_num >= high:
            #     p.console.print(f"Skipping {task_id} as it is not in {id_range}")
            #     continue
            if id_num not in id_range:
                p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                continue
                    
            p.console.print(f"Processing {task_id}")
            
            # Load model outputs
            model_outputs = load_model_outputs(args.model_output_dir, task_id)
            if not model_outputs:
                p.console.print(f"No model outputs found for {task_id}, skipping")
                continue
                
            # Generate adversarial prompt with retries
            max_tries = 1
            tries = 0
            success = False
            
            while tries < max_tries and not success:
                try:
                    new_prompt = generate_adversarial_prompt(task["prompt"], model_outputs, claude)
                    print("claude output: \n", new_prompt)
                    # Extract and evaluate reference solution
                    problem_description = extract_problem_description(new_prompt)
                    new_prompt_solution = answer_question_via_claude(problem_description, claude)
                    print("new_prompt_solution: \n", new_prompt_solution)
                    #ref_solution = extract_solution(new_prompt)
                    ref_solution = extract_solution(new_prompt_solution)
                    
                    print("problem_description: \n", problem_description)
                    print("ref_solution: \n", ref_solution)
                    
                    if ref_solution and evaluate_solution(task_id, ref_solution, args.dataset):
                        success = True
                        # Save successful prompt
                        os.makedirs(args.output_dir, exist_ok=True)
                        out_path = os.path.join(args.output_dir, f"{task_id.replace('/', '_')}.jsonl")
                        print("out_path: ", out_path)
                        #os.makedirs(out_path, exist_ok=True)
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(json.dumps({"problem_description": problem_description, "ref_solution": ref_solution, "claude_output": new_prompt}))
                    else:
                        p.console.print(f"Generated solution failed tests, retrying ({tries + 1}/{max_tries})")
                        tries += 1
                        
                except Exception as e:
                    p.console.print(f"Error processing {task_id}: {str(e)}")
                    tries += 1
            
            if not success:
                p.console.print(f"Failed to generate valid solution for {task_id} after {max_tries} attempts")
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, choices=["humaneval", "mbpp"])
    parser.add_argument("--model_output_dir", required=True, type=str, 
                      help="Directory containing model outputs")
    parser.add_argument("--output_dir", required=True, type=str,
                      help="Directory to save adversarial prompts")
    parser.add_argument("--claude_key", required=False, type=str, default=None,
                      help="Anthropic API key")
    parser.add_argument("--google_key", required=False, type=str, default=None,
                      help="Google API key")
    parser.add_argument("--openai_key", required=False, type=str, default=None,
                      help="Anthropic API key")
    # parser.add_argument("--id-range", nargs=2, type=int, default=None,
    #                   help="Optional range of problem IDs to process")
    
    args = parser.parse_args()
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = args.google_key
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = args.claude_key
    adversarial_generate(args)

if __name__ == "__main__":
    main()



    # context = f'''You are an expert computer science professor who specializes in identifying students' misconceptions and creating targeted practice problems. I will provide you with:

    # An original programming problem {original_prompt}
    # A solution attempt from an LLM {model_outputs}

    # Your task is to generate an adversarial programming problem that reveals potential flaws or shortcuts in the LLM's solution approach while maintaining the core mathematical/computational requirements of the original problem.
    # Follow these steps in your analysis and generation:

    # SOLUTION ANALYSIS
    # First, analyze the provided LLM solution for:


    # Potential algorithmic shortcuts or assumptions
    # Edge cases that might be missed
    # Implementation patterns that suggest misunderstanding of the core problem
    # Over-reliance on specific data structures or algorithms that may not be optimal
    # Handling of special cases or boundary conditions


    # IDENTIFY EXPLOITATION OPPORTUNITIES
    # Based on your analysis, identify:


    # Which assumptions in the solution could be challenged
    # What edge cases could reveal flaws in the approach
    # How the solution might fail while still passing the original test cases
    # What problem-solving patterns the solution overly relies on


    # GENERATE NEW PROBLEM AND CORRECT SOLUTION
    # Create a new problem and its correct solution that:
    # a) Maintains Mathematical Identity:

    # The fundamental computational task must remain identical
    # All original test cases must remain valid
    # The input/output contract must remain unchanged



    # b) Changes Problem Expression:

    # Use different domain terminology and context
    # Avoid keywords that might trigger specific solution patterns
    # Frame the problem in a way that makes flawed approaches less obvious
    # Use domain-specific language that discourages problematic shortcuts

    # c) Crafts Strategic Test Cases:

    # Include test cases that specifically target identified weaknesses
    # Ensure test cases pass with a correct solution but fail with the flawed approach
    # Maintain similar input sizes and complexity as original test cases
    

    # d) Provides Correct Implementation:

    # Implement a solution that handles all edge cases
    # Include detailed comments explaining key implementation decisions
    # Demonstrate why this approach is more robust than the flawed solution


    # OUTPUT FORMAT
    # Present your response in this structured format:

    # <new_problem>
    # (Insert your generated problem text here, including:

    # Function signature
    # Docstring with problem description
    # Example test cases that reveal solution flaws)
    # </new_problem>

    # EXPLANATION:

    # Key flaws in original solution:

    # List identified shortcomings


    # Adversarial design choices:

    # Explain how new problem targets each flaw
    # Describe how context/terminology changes discourage shortcuts


    # Test case analysis:

    # Explain why new test cases would fail with flawed solution
    # Demonstrate mathematical equivalence to original problem


    # Solution design decisions:

    # Explain why the provided solution is more robust
    # Highlight how it handles edge cases that would trip up the flawed approach
    # Describe any optimizations or improvements over the original solution



    # Remember:

    # The new problem must be mathematically identical to the original
    # All changes should serve to reveal potential flaws in the solution approach
    # The problem should remain clear and well-defined
    # Test cases should be carefully crafted to expose shortcuts while remaining valid
    # The provided solution should be optimal and handle all edge cases and import necessary libraries'''
