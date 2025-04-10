import numpy as np 
import pandas as pd
from typing import Union, List, Literal
import json
import itertools
import argparse
import logging
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument("--junit_test_path", type=str, default="out/completions/gemini-2.0-flash-thinking-exp-01-21/cot/pass1/2025-04-09_09-34-57/junit_results.jsonl", help="Model's HF directory or local path")
    parser.add_argument("--out_dir", type=str, default="./out", help="Outputs directory")
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


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG,
        datefmt="%m/%d/%Y %H:%M:%S",
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        filename="eval_metrics.log",
        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    args = parse_arguments()
    model_name = args.junit_test_path.split("/")[2]
    mode = args.junit_test_path.split("/")[3]
    k = args.k

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
    logger.info(f"First: {len(grouped_list)}")

    num_samples = [k] * len(grouped_list)
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
    pass_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime, k, mode="at")) * 100, 1), ".1f")
    pass_k_mandatory = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime_mandatory, k, mode="at")) * 100, 1), ".1f")

    # pass^K       
    # compilation_power_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_compilations, 2, mode="power")) * 100, 1), ".1f")
    # pass_power_k = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime, 2, mode="power")) * 100, 1), ".1f")
    # pass_power_k_mandatory = format(round(np.mean(estimate_pass_k(num_samples, num_correct_runtime_mandatory, 2, mode="power")) * 100, 1), ".1f")


    logger.info("-"*25)
    logger.info(f"Mean compilation@{k}: {compilation_k}")
    logger.info(f"Mean pass@{k} - SOFT: {pass_k_mandatory}")
    logger.info(f"Mean pass@{k} - HARD: {pass_k}")
    logger.info("-"*25)
    
    # logger.info(f"Mean compilation^{2}: {compilation_power_k}")
    # logger.info(f"Mean pass^{2} - SOFT: {pass_power_k_mandatory}")
    # logger.info(f"Mean pass^{2} - HARD: {pass_power_k}")
    # logger.info("-"*25)

    print(all_grades[:10])
    
