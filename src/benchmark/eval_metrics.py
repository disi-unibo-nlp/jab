import numpy as np 
import pandas as pd
from typing import Union, List
import json
import itertools
import argparse
import logging
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument("--junit_test_path", type=str, default="out/completions/Qwen2.5-Coder-7B-Instruct/cot/pass10/2025-04-03_11-10-59/junit_results_2.jsonl", help="Model's HF directory or local path")
    parser.add_argument("--out_dir", type=str, default="./out", help="Outputs directory")
    parser.add_argument("--k", type=int, default=10, help="value of K in Pass@k")

    return parser.parse_args()

def estimate_pass_at_k(
    num_samples: Union[int, List[int]],
    num_correct: Union[List[int]],
    k: int
):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

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

    num_samples = [k] * len(grouped_list)
    num_correct_compilations = []
    num_correct_runtime = []
    num_correct_runtime_mandatory = []

    for exam_session in grouped_list:
        session_attempts = exam_session['attempts']
        compilation_passed = [attempt for attempt in session_attempts if attempt['compilation_passed']]
        runtime_passed = [attempt for attempt in session_attempts if attempt['runtime_passed']]
        runtime_passed_mandatory = [attempt for attempt in session_attempts if attempt['runtime_passed_mandatory']]
        num_correct_compilations.append(len(compilation_passed))
        num_correct_runtime.append(len(runtime_passed))
        num_correct_runtime_mandatory.append(len(runtime_passed_mandatory))
    
    compilation_k = format(round(np.mean(estimate_pass_at_k(num_samples, num_correct_compilations, k)) * 100, 1), ".1f")
    pass_k = format(round(np.mean(estimate_pass_at_k(num_samples, num_correct_runtime, k)) * 100, 1), ".1f")
    pass_k_mandatory = format(round(np.mean(estimate_pass_at_k(num_samples, num_correct_runtime_mandatory, k)) * 100, 1), ".1f")

    logger.info("-"*25)
    logger.info(f"Mean compilation@{k}: {compilation_k}")
    logger.info(f"Mean pass@{k} - SOFT: {pass_k_mandatory}")
    logger.info(f"Mean pass@{k} - HARD: {pass_k}")
    logger.info("-"*25)
    
