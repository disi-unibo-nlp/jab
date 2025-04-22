import numpy as np 
import pandas as pd
from typing import Union, List, Literal
import json
import itertools
import argparse
import logging
import os
from collections import defaultdict


# out/completions/gemini-2.0-flash-thinking-exp-01-21/agent/pass1/2025-04-11_08-29-16/completions_agent.jsonl
# out/completions/gemini-2.0-flash-thinking-exp-01-21/cot/pass1/2025-04-09_09-34-57/junit_results.jsonl
# out/completions/gemini-2.0-flash-thinking-exp-01-21/cot/pass1/2025-04-15_08-56-40/completions_cot.jsonl
# out/completions/Qwen2.5-Coder-7B-Instruct/cot/pass10/2025-04-03_11-10-59/junit_results.jsonl
# out/completions/gemini-2.5-pro-exp-03-25/cot/pass1/2025-04-15_20-21-11/junit_results.jsonl
# out/completions/gemini-2.0-flash-thinking-exp-01-21/cot/pass1/2025-04-16_09-19-20/unittest_cot_python.jsonl

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument("--junit_test_path", type=str, default="out/completions/Qwen2.5-Coder-7B-Instruct/agent/n1/2025-04-22_21-54-40/completions_agent_python.jsonl", help="Model's HF directory or local path")
    parser.add_argument("--out_dir", type=str, default="./eval_metrics", help="Outputs directory")
    parser.add_argument("--k", type=int, default=1, help="value of K in Pass@k")
    parser.add_argument("--max_score", type=int, default=14, help="max reachable score in the written part of the exam.")


    return parser.parse_args()

# def estimate_pass_at_k(
#     num_samples: Union[int, List[int]],
#     num_correct: Union[List[int]],
#     k: int
# ):
#     """
#     Estimates pass@k of each problem and returns them in an array.
#     """

#     def estimator(n: int, c: int, k: int) -> float:
#         """
#         Calculates 1 - comb(n - c, k) / comb(n, k).
#         """
#         if n - c < k:
#             return 1.0
#         return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

#     if isinstance(num_samples, int):
#         num_samples_it = itertools.repeat(num_samples, len(num_correct))
#     else:
#         assert len(num_samples) == len(num_correct)
#         num_samples_it = iter(num_samples)

#     return [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]


def estimate_pass_k(
    num_samples: Union[int, List[int]],
    num_correct: List[int],
    k: int,
    mode: Literal["at", "power"] = "at"
):
    """
    Estimates pass@k or pass^k of each problem and returns them in an array.

    Args:
        num_samples: Total number of attempts (n) per problem (single int or list).
        num_correct: List of number of correct solutions (c) for each problem.
        k: Number of independent trials.
        mode: "at" for pass@k, "power" for pass^k.

    Returns:
        A list of floats representing pass@k or pass^k for each problem.
    """

    def pass_at_k(n: int, c: int, k: int) -> float:
        """
        Calculates pass@k = 1 - comb(n - c, k) / comb(n, k)
        using a numerically stable product form.
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def pass_power_k(n: int, c: int, k: int) -> float:
        """
        Calculates pass^k = (c / n) ** k
        """
        if n == 0:
            return 0.0
        return (c / n) ** k

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    estimator = pass_at_k if mode == "at" else pass_power_k
    return [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]

def calculate_scores(grouped_list, filter_by_year=None):

    if filter_by_year:
        grouped_list = [el for el in grouped_list if str(year) in el['id']]
        logger.info(f"\n")
        logger.info(f"Number of sessions: {len(grouped_list)}")

    num_samples = [len(el['attempts']) for el in grouped_list] #* len(grouped_list)
    print(f"Num samples: {num_samples}")
    num_correct_compilations = []
    num_correct_runtime = []
    num_correct_runtime_mandatory = []
    all_grades = []
    max_score = args.max_score
    for exam_session in grouped_list:
        session_attempts = exam_session['attempts']
        compilation_passed = [attempt for attempt in session_attempts if attempt['compilation_passed']]
        runtime_passed = [attempt for attempt in session_attempts if attempt['runtime_passed']]
        runtime_passed_mandatory = [attempt for attempt in session_attempts if attempt['runtime_passed_mandatory']]
        error_details = [attempt['test_details'] for attempt in session_attempts ]
        num_correct_compilations.append(len(compilation_passed))
        num_correct_runtime.append(len(runtime_passed))
        num_correct_runtime_mandatory.append(len(runtime_passed_mandatory))

        for err_details in error_details:
            if err_details:
                num_tests = err_details['tests_found']
                test_success = err_details["tests_successful"]

                if num_tests == test_success:
                    final_grade = max_score
                else:
                    point_per_test = max_score / num_tests
                    final_grade = round(test_success * point_per_test)

                all_grades.append({"id": exam_session['id'], "grade": final_grade})
            
            else:
                # compilation error
                all_grades.append({"id": exam_session['id'], "grade": 0})

    # pass@K       
    compilation_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_compilations, k, mode="at")) * 100, 1), ".1f")
    pass_k_mandatory = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime_mandatory, k, mode="at")) * 100, 1), ".1f")
    pass_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime, k, mode="at")) * 100, 1), ".1f")

    return compilation_k, pass_k_mandatory, pass_k


if __name__ == "__main__":

    args = parse_arguments()
    model_name = args.junit_test_path.split("/")[2]
    out_dir = f"{args.out_dir}/{model_name}/pass{args.k}"
    os.makedirs(out_dir, exist_ok=True)

    model_name = args.junit_test_path.split("/")[2]
    mode = args.junit_test_path.split("/")[3]
    k = args.k
    code_language = "java" if "python" not in args.junit_test_path else "python"

    logging.basicConfig(level=logging.DEBUG,
        datefmt="%m/%d/%Y %H:%M:%S",
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        filename=f"{out_dir}/eval_metrics_{mode}_{code_language}.log",
        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())


    with open(args.junit_test_path) as f:
        results = [json.loads(line) for line in f.readlines()]
        results = [{"id": f"{el['year']}_{el['session']}", **el} for el in results]
    
    # Group by 'id'
    grouped_data = defaultdict(list)
    for item in results:
        grouped_data[item["id"]].append(item)

    # Convert to list of dictionaries
    grouped_list = [{"id": id_exam, "attempts": v} for id_exam, v in grouped_data.items()]

    logger.info(f"Number of exam session: {len(grouped_list)}")

    compilation_k, pass_k_mandatory, pass_k = calculate_scores(grouped_list)

    logger.info("-"*25)
    logger.info(f"Mean compilation@{k}: {compilation_k}")
    logger.info(f"Mean pass@{k} - SOFT: {pass_k_mandatory}")
    logger.info(f"Mean pass@{k} - HARD: {pass_k}")
    logger.info("-"*25)
    logger.info("\n")

    for year in range(2014, 2025):
        compilation_k, pass_k_mandatory, pass_k = calculate_scores(grouped_list, filter_by_year=str(year))
        
        logger.info("-"*25)
        logger.info(f"YEAR: {year}")
        logger.info("-"*25)
        logger.info(f"Mean compilation@{k}: {compilation_k}")
        logger.info(f"Mean pass@{k} - SOFT: {pass_k_mandatory}")
        logger.info(f"Mean pass@{k} - HARD: {pass_k}")
        
        

    #logger.info(f"First: {len(grouped_list)}")

    # num_samples = [k] * len(grouped_list)
    # num_correct_compilations = []
    # num_correct_runtime = []
    # num_correct_runtime_mandatory = []
    # all_grades = []
    # max_score = args.max_score
    # for exam_session in grouped_list:
    #     session_attempts = exam_session['attempts']
    #     compilation_passed = [attempt for attempt in session_attempts if attempt['compilation_passed']]
    #     runtime_passed = [attempt for attempt in session_attempts if attempt['runtime_passed']]
    #     runtime_passed_mandatory = [attempt for attempt in session_attempts if attempt['runtime_passed_mandatory']]
    #     error_details = [attempt['test_details'] for attempt in session_attempts ]
    #     num_correct_compilations.append(len(compilation_passed))
    #     num_correct_runtime.append(len(runtime_passed))
    #     num_correct_runtime_mandatory.append(len(runtime_passed_mandatory))

    #     for err_details in error_details:
    #         if err_details:
    #             num_tests = err_details['tests_found']
    #             test_success = err_details["tests_successful"]

    #             if num_tests == test_success:
    #                 final_grade = max_score
    #             else:
    #                 point_per_test = max_score / num_tests
    #                 final_grade = round(test_success * point_per_test)

    #             all_grades.append({"id": exam_session['id'], "grade": final_grade})
            
    #         else:
    #             # compilation error
    #             all_grades.append({"id": exam_session['id'], "grade": 0})

    # # pass@K       
    # compilation_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_compilations, k, mode="at")) * 100, 1), ".1f")
    # pass_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime, k, mode="at")) * 100, 1), ".1f")
    # pass_k_mandatory = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime_mandatory, k, mode="at")) * 100, 1), ".1f")

    # pass^K       
    # compilation_power_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_compilations, 2, mode="power")) * 100, 1), ".1f")
    # pass_power_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime, 2, mode="power")) * 100, 1), ".1f")
    # pass_power_k_mandatory = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime_mandatory, 2, mode="power")) * 100, 1), ".1f")


    
    
    # logger.info(f"Mean compilation^{2}: {compilation_power_k}")
    # logger.info(f"Mean pass^{2} - SOFT: {pass_power_k_mandatory}")
    # logger.info(f"Mean pass^{2} - HARD: {pass_power_k}")
    # logger.info("-"*25)

    #print(all_grades[:10])
    
