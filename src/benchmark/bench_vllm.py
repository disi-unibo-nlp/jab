import torch
import json
import os
import logging
import pandas as pd
import numpy as np 
import json
import argparse
import subprocess
import time
import re

from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from src.benchmark.utils import check_mandatory_tests, extract_python_code, exec_python_tests_and_parse
# Load variables from the .env file
load_dotenv()

# Manually set the required environment variable
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument("--model_path", type=str, default="open-r1/OlympicCoder-7B", help="Model's HF directory or local path")
    parser.add_argument("--dataset_path", type=str, default="disi-unibo-nlp/JAB", help="Dataset HF directory")
    parser.add_argument("--out_dir", type=str, default="./out", help="Outputs directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of data to process in train set. Default is -1 to process all data.")
    parser.add_argument("--start_idx", type=int, default=0, help="Index of first prompt to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Maximum number of data to process per batch.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory to store model weights")
    parser.add_argument("--max_model_len", type=int, default=8000, help="Maximum input sequence length")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p sampling.")
    parser.add_argument("--n_out_sequences", type=int, default=1, help="Number of generated sequences per instance")
    parser.add_argument("--n_sampling", type=int, default=1, help="Number of solutions to generate for a given prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature parameter")
    parser.add_argument("--mode", type=str, choices=["cot", "agent"], default='cot', help="Inference mode: CoT or Agent")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use for inference.")
    parser.add_argument("--n_rounds", type=int, default=3, help="Number of rounds to use for inference.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max number of tokens to generate in CoT prompting.")
    parser.add_argument("--year_sessions", type=str, default="", help="Specific years to consider, separated by comma. E.g: 2016,2018,2021")
    parser.add_argument("--junit_jar", default= "lib/junit-platform-console-standalone-1.13.0-M2.jar", help="Path to the JUnit standalone JAR file.")

    return parser.parse_args()


def create_exams(dataset, exams_dir="exams"):

    for item in tqdm(dataset, desc="Creating exams"):
        year = item['year']
        year_dir = f"oop{year}"
        session_dir = item['session']
        solution_dir = f"{args.out_dir}/{exams_dir}/{MODEL_NAME}/{args.mode}/{now_dir}/{year_dir}/{session_dir}/sol1"

        # solution_dir = f"{args.out_dir}/{exams_dir}/{MODEL_NAME}/{now_dir}/{year_dir}/{session_dir}/sol1"

        range_to_consider = range(args.n_out_sequences) if args.mode != "agent" else range(args.n_sampling)
        for k in range_to_consider:
            # create utils
            for util_class in item['utility_classes']:
                util_class_filename = util_class['filename']
                new_folder = f"k{k}" if len(range_to_consider) > 1 else "pass1"
                actual_sol_dir = solution_dir + f"/{new_folder}"
                os.makedirs(actual_sol_dir, exist_ok=True)
                with open(f"{actual_sol_dir}/{util_class_filename}", "w") as f:
                    content = util_class["content"].strip()

                    # Replace old suffixes with the new folder structure
                    content = (content
                        .replace(".e1;", ".sol1;")
                        .replace(".sol2;", ".sol1;")
                        .replace(".e2;", ".sol1;")
                        .replace(".sol1;", f".sol1.{new_folder};")
                        .replace(".sol1.Evaluation", f".sol1.{new_folder}.Evaluation")
                        .replace("import static ex2015.a01a.sol1.", f"import static ex2015.a01a.sol1.{new_folder}.")
                    )

                    # Write the modified content back to the file
                    f.write(content)
            # create test
            test_filename = item['test']['filename']

            with open(f'{actual_sol_dir}/{test_filename}', 'w') as f:
                test_content = item['test']['content'].strip()

                # Custom replacements in your script
                test_content = (
                    test_content.replace('.e1;', '.sol1;')
                                .replace('.sol2;', '.sol1;')
                                .replace('.e2;', '.sol1;')
                                .replace('.sol1.', f'.sol1.{new_folder}.')
                                .replace('.sol1;', f'.sol1.{new_folder};')
                                #.replace('.sol1.Evaluation', f'.sol1.{new_folder}.Evaluation')
                )

                f.write(test_content)
            
            # with open(f'{actual_sol_dir}/{test_filename}', 'w') as f:
            #     test_content = item['test']['content'].strip()

            #     # correct specific human errors of mispelling
            #     if year == "2020" and session_dir == "a05":
            #         test_content = test_content.replace("createRechargableBattery", "createRechargeableBattery")
            #         test_content = test_content.replace("createSecureAndRechargableBattery", "createSecureAndRechargeableBattery")

            #     f.write(test_content.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;').replace('.sol1;', f'.sol1.{new_folder};').replace('.sol1.Evaluation',f'.sol1.{new_folder}.Evaluation'))

def compile_exam(year, session, exams_dir="exams"):

    compile_errors = []
    
        
    year_dir = f"oop{year}"
    session_dir = session
    
    solution_dir = f"{args.out_dir}/{exams_dir}/{MODEL_NAME}/{now_dir}/{year_dir}/{session_dir}/sol1"
    range_to_consider = range(args.n_out_sequences) if args.mode != "agent" else range(args.n_sampling)

    for k in range_to_consider:
        # create utils
        for util_class in item['utility_classes']:
            util_class_filename = util_class['filename']
            actual_sol_dir = solution_dir + (f"/k{k}" if len(range_to_consider) > 1 else "/pass1")

            bin_path = os.path.join(actual_sol_dir, "bin")

            os.makedirs(bin_path, exist_ok=True)

            for root, _, files in os.walk(bin_path):
                for file in files:
                    os.remove(os.path.join(root, file))

        java_files = sorted([os.path.join(actual_sol_dir, f) for f in os.listdir(actual_sol_dir) if f.endswith(".java") and not f.startswith("Test")])
        
        if not java_files:
            logger.warning(f"No Java files found in {actual_sol_dir}. Skipping.")
            continue

        logger.info(f"Compiling {len(java_files)} files in {actual_sol_dir}...")
        compile_command = ["javac", "-cp", JUNIT_JAR, "-d", bin_path] + java_files

        try:
            subprocess.run(compile_command, check=True, text=True, capture_output=True)
            #safe_sessions.append({"year": year, "session": session})
        except subprocess.CalledProcessError as e:
            err_output = e.stderr.strip().replace("\n", " ")
            logger.error(f"Compilation failed for {actual_sol_dir}: {err_output}")
            compile_errors.append({"year": year, "session": session, "err": err_output})
            continue

    return compile_errors


# def exec_java_code(java_codes, year, session, k=None, exams_dir="exams"):

#     compile_errors = []
#     runtime_errors = []

#     year_dir = f"oop{year}"
#     session_dir = session
    
#     solution_dir = f"{args.out_dir}/{exams_dir}/{MODEL_NAME}/{now_dir}/{year_dir}/{session_dir}/sol1"
#     new_folder = f"k{k}" if not k is None else "pass1"
#     actual_sol_dir = solution_dir + f"/{new_folder}"

#     bin_path = os.path.join(actual_sol_dir, "bin")

#     os.makedirs(bin_path, exist_ok=True)

#     for root, _, files in os.walk(bin_path):
#         for file in files:
#             os.remove(os.path.join(root, file))
    
#     for k, code in enumerate(java_codes):
#         code_filename = extract_filename(code)
#         if code_filename and code_filename != "Test":
#             logger.info(f"Filename {k}: {code_filename}")
#             code = code.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;').replace('.sol1;', f'.sol1.{new_folder};').replace('.sol1.Evaluation',f'.sol1.{new_folder}.Evaluation')
#             with open(os.path.join(actual_sol_dir, code_filename + ".java"), 'w') as f:
#                 f.write(code)
#         else:
#             logger.info(f"This code is not a file to consider:\n\n{code}")

#     java_files = sorted([os.path.join(actual_sol_dir, f) for f in os.listdir(actual_sol_dir) if f.endswith(".java")])
    
#     if not java_files:
#         logger.warning(f"No Java files found in {actual_sol_dir}. Skipping.")
#         return

#     logger.info(f"Compiling {len(java_files)} files in {actual_sol_dir}...")
#     compile_command = ["javac", "-cp", JUNIT_JAR, "-d", bin_path] + java_files

#     try:
#         subprocess.run(compile_command, check=True, text=True, capture_output=True)
#     except subprocess.CalledProcessError as e:
#         err_output = e.stderr.strip().replace("\n", " ")
#         logger.error(f"Compilation failed for {actual_sol_dir}: {err_output}")
#         compile_errors.append({"year": year, "session": session, "error": err_output})

#     if not compile_errors:

#         logger.info(f"Running tests for {actual_sol_dir}...")
#         run_command = [
#             "java", "-cp", f"{bin_path}:{JUNIT_JAR}",
#             "org.junit.platform.console.ConsoleLauncher",
#             "--class-path", bin_path,
#             "--scan-class-path"
#         ]

#         try:
#             result = subprocess.run(run_command, capture_output=True, text=True, timeout=10)
#             logger.info(result.stdout)

#             # If the command runs but tests fail, check stderr for failures
#             if result.returncode != 0:
#                 logger.info(f"{actual_sol_dir}: Tests failed with errors.")
                
#                 runtime_errors.append({
#                     "year": year,
#                     "session": session,
#                     "error": "Failures " + result.stdout.split("Failures")[1].strip()
#                 })

#         except subprocess.SubprocessError as e:
#             logger.info(f"{actual_sol_dir}: failed with error {e}\n")
#             runtime_errors.append({
#                 {"year": year, "session": session, "error": result.stderr if 'result' in locals() else str(e)}
#             })
#         except TimeoutError:
#             logger.info(f"{actual_sol_dir}: timed out after 10 seconds\n")
#             runtime_errors.append({
#                 {"year": year, "session": session, "error": "Timeout error."}
#             })
    
#     return compile_errors, runtime_errors

def extract_failed_test_methods(stdout):
    """
    Extracts the names of the failed test methods from the JUnit output.
    """
    lines = stdout.split('\n')
    failed_methods = []
    for line in lines:
        if 'MethodSource' in line and 'methodName' in line:
            start = line.find("methodName = '") + len("methodName = '")
            end = line.find("'", start)
            if start > 0 and end > start:
                failed_methods.append(line[start:end])
    return failed_methods          

def exec_java_code(java_files, year, session, now_dir, k=None, exams_dir="exams"):
    compile_errors = []
    runtime_errors = []

    year_dir = f"oop{year}"
    session_dir = session
   
    solution_dir = f"{args.out_dir}/{exams_dir}/{MODEL_NAME}/{args.mode}/{now_dir}/{year_dir}/{session_dir}/sol1"
    new_folder = f"k{k}" if not k is None else "pass1"
    actual_sol_dir = solution_dir + f"/{new_folder}"

    bin_path = os.path.join(actual_sol_dir, "bin")

    os.makedirs(bin_path, exist_ok=True)

    for root, _, files in os.walk(bin_path):
        for file in files:
            os.remove(os.path.join(root, file))

    for k_code, code in enumerate(java_files):
        code_filename = extract_filename(code)
        if code_filename and code_filename != "Test":
            logger.info(f"Filename {k_code}: {code_filename}")
            code = code.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;').replace('.sol1.',f'.sol1.{new_folder}.').replace('.sol1;', f'.sol1.{new_folder};')
            with open(os.path.join(actual_sol_dir, code_filename + ".java"), 'w') as f:
                f.write(code)
        else:
            logger.info(f"This code is not a file to consider:\n\n{code}")

    java_files = sorted([os.path.join(actual_sol_dir, f) for f in os.listdir(actual_sol_dir) if f.endswith(".java")])
    
    if not java_files:
        logger.warning(f"No Java files found in {actual_sol_dir}. Skipping.")
        return

    JUNIT_JAR = args.junit_jar
    logger.info(f"Compiling {len(java_files)} files in {actual_sol_dir}...")
    compile_command = ["javac", "-cp", JUNIT_JAR, "-d", bin_path] + java_files
    test_details = {}

    try:
        subprocess.run(compile_command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err_output = e.stderr.strip().replace("\n", " ")
        logger.error(f"Compilation failed for {actual_sol_dir}: {err_output}")
        compile_errors.append({"error": err_output})

    if not compile_errors:

        logger.info(f"Running tests for {actual_sol_dir}...")
        run_command = [
            "java", "-cp", f"{bin_path}:{JUNIT_JAR}",
            "org.junit.platform.console.ConsoleLauncher",
            "--class-path", bin_path,
            "--scan-class-path"
        ]

        try:
            result = subprocess.run(run_command, capture_output=True, text=True, timeout=10)
            logger.info(result.stdout)

            test_details = parse_test_output(result.stdout)
            # If the command runs but tests fail, check stderr for failures
            if result.returncode != 0:
                logger.info(f"{actual_sol_dir}: Tests failed with errors.")
                failed_methods = extract_failed_test_methods(result.stdout)
                runtime_errors.append({
                    "error": "Failures " + result.stdout.split("Failures")[1].strip(),
                    "fails": failed_methods
                })

        except subprocess.SubprocessError as e:
            logger.info(f"{actual_sol_dir}: failed with error {e}\n")
            runtime_errors.append({
                "error": result.stderr if 'result' in locals() else str(e)
            })
        except TimeoutError:
            logger.info(f"{actual_sol_dir}: timed out after 10 seconds\n")
            runtime_errors.append({
                "error": "Timeout error."
            })

    return compile_errors, runtime_errors, test_details

def parse_test_output(output):
    details = {
        "containers_found": 0,
        "containers_skipped": 0,
        "containers_started": 0,
        "containers_aborted": 0,
        "containers_successful": 0,
        "containers_failed": 0,
        "tests_found": 0,
        "tests_skipped": 0,
        "tests_started": 0,
        "tests_aborted": 0,
        "tests_successful": 0,
        "tests_failed": 0
    }
    
    for line in output.splitlines():
        if "containers found" in line:
            details["containers_found"] = int(line.split()[1])
        elif "containers skipped" in line:
            details["containers_skipped"] = int(line.split()[1])
        elif "containers started" in line:
            details["containers_started"] = int(line.split()[1])
        elif "containers aborted" in line:
            details["containers_aborted"] = int(line.split()[1])
        elif "containers successful" in line:
            details["containers_successful"] = int(line.split()[1])
        elif "tests found" in line:
            details["tests_found"] = int(line.split()[1])
        elif "tests skipped" in line:
            details["tests_skipped"] = int(line.split()[1])
        elif "tests started" in line:
            details["tests_started"] = int(line.split()[1])
        elif "tests aborted" in line:
            details["tests_aborted"] = int(line.split()[1])
        elif "tests successful" in line:
            details["tests_successful"] = int(line.split()[1])
        elif "tests failed" in line:
            details["tests_failed"] = int(line.split()[1])
        
    return details

def extract_and_remove_package_line(java_code):
    match = re.search(r'^package\s+[^\n]+;', java_code, flags=re.MULTILINE)
    package_line = match.group(0) if match else None
    cleaned_code = re.sub(r'^package\s+[^\n]+;\n?', '', java_code, flags=re.MULTILINE)
    return package_line, cleaned_code

def extract_java_code(text):
    """Extracts Java code blocks enclosed between ```java and ```."""
    pattern = r"```java\s+([\s\S]*?)\s+```"
    matches = [match.strip() for match in re.findall(pattern, text)]
    return matches if len(matches) > 0 else ""

def extract_filename(java_code):
    # Matches class, interface, enum, annotation, and record declarations
    pattern = r"(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?(?:class|interface|enum|@interface|record)\s+([A-Za-z_][A-Za-z0-9_]*)"

    match = re.search(pattern, java_code)
    return match.group(1) if match else None

if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="output_single_proc.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    args = parse_arguments()
    print(args)

    JUNIT_JAR = args.junit_jar
    MODEL_NAME = args.model_path.split("/")[-1]

    if args.n_gpus > 1: 
        import ray
        ray.init(_temp_dir="/my_local_tmp_dir", log_to_driver=False)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    sampling_params = SamplingParams(
        n=args.n_out_sequences, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_tokens, 
        seed=None if (args.n_out_sequences > 1 and args.mode == "cot") or (args.n_sampling > 1 and args.mode == "agent") else 0
    )

    
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in args.model_path.lower() else "auto",
        quantization="awq_marlin" if "awq" in args.model_path.lower() else None,
        enforce_eager=True if "33b" in args.model_path.lower() or "32b" in args.model_path.lower() else False,
        max_model_len=args.max_model_len if args.max_model_len > 0 else None,
        trust_remote_code=True,
        tensor_parallel_size=args.n_gpus,
    )

    dataset = load_dataset(args.dataset_path, split="test")

    with open('data/optional_conditions.json') as f:
        optional_conditions = json.load(f)

    if args.year_sessions: 
        years_to_consider = args.year_sessions.split(",")
        years_to_consider = [int(el) for el in years_to_consider]
        dataset = dataset.filter(lambda example: example['id'] in years_to_consider)
    
    if args.start_idx > 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, len(dataset)))
    
    if args.max_samples > 0: # to use for debug
        dataset = dataset.select(range(args.max_samples))

    
    if args.mode == "agent" and "python" not in args.dataset_path:

        logger.info("Creating exams...")
        create_exams(dataset)
        logger.info("Done!")

        # logger.info("Compiling exams utilities...")
        # safe_compilation = True
        # for item in dataset:

        #     compile_errors = compile_exam(year=item['year'], session=item['session'])
        #     if compile_errors:
        #         logger.info(f"Errors occurs during compilation of exam `{item['year']}_{item['session']}`. You need to solve such errors before you can run the code. ")
        #         safe_compilation = False
        #         break

        # if safe_compilation:      
        #     logger.info("Done.")

    
    SYS_INSTRUCTION = """You are an expert Java developer. Your task is to solve the given university-level Java exam in a single attempt, ensuring that your solution correctly passes all JUnit tests defined in the provided `Test.java` file.  

You may use the provided utility Java files as needed. Your final answer must consist of one or more Java scripts, each enclosed separately between ```java and ``` in different code blocks."""
    
    if "python" in args.dataset_path:
        SYS_INSTRUCTION = """You are an expert Python developer. Your task is to solve the given university-level Python exam in a single attempt, ensuring that your solution correctly passes all unit tests defined in the provided `test.py` script.  

You may use the provided utilities as needed. Your final answer must consist of one Python script including only the solution to the problem, enclosed between ```python and ```."""

    prompts = []
    for i, item in enumerate(dataset):
        package_line = ""
        if not "python" in args.dataset_path:
            PROMPT_TEMPLATE = """### Utility Files:\n\n{utility_files}\n\n### Test File:\n```java\n\n{test_file}```"""
            
            utilities = item['utility_classes']

            test_content = item['test']['content']

            # # correct specific human errors of mispelling on specific test data
            # if item['year'] == "2020" and item['session'] == "a05":
            #     test_content = test_content.replace("createRechargableBattery", "createRechargeableBattery")
            #     test_content = test_content.replace("createSecureAndRechargableBattery", "createSecureAndRechargeableBattery")

            # # fix consistency in package path
            # test_content = test_content.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;')
            
            package_line, test_content = extract_and_remove_package_line(java_code=test_content)
            test_file = f"// {item['test']['filename']}\n\n{test_content}"

            utility_files = ""
            for util_class in utilities:
                # fix consistency in package path
                _, util_class_content = extract_and_remove_package_line(util_class['content'].strip())#.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;').strip()
                utility_files += f"```java\n// {util_class['filename']}\n\n{util_class_content}```\n"

            prompt = PROMPT_TEMPLATE.format(utility_files=utility_files, test_file=test_file)
        else:
            PROMPT_TEMPLATE = """```python\n\n### Utility\n\n{utility}\n\n### Test\n\n{test}```"""
            utilities = item['utility_classes']
            test_content = item['test']['content']
            prompt = PROMPT_TEMPLATE.format(utility=utilities, test=test_content)

        

        if "OlympicCoder" in args.model_path:
            
            messages = [
                {"role": "user", "content": SYS_INSTRUCTION.replace('You are an expert Java developer.', '').strip() + "\n\n" + prompt}
            ]
        
        if "Qwen2.5-Coder" in args.model_path:
            
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": SYS_INSTRUCTION.replace('You are an expert Java developer.', '').strip() + "\n\n" + prompt}
            ]
        if "deepseek-coder" in args.model_path:
            messages = [
                {"role": "user", "content": SYS_INSTRUCTION.replace('You are an expert Java developer.', '').strip() + "\n\n" + prompt}
            ]

        if "phi-4" in args.model_path.lower():
            messages = [
                {"role": "system", "content": SYS_INSTRUCTION},
                {"role": "user", "content": prompt}
            ]
            
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append({
            "id": f"oop{item['year']}_{item['session']}", 
            "prompt": text.strip(), 
            "package": package_line.strip(),
            "chat_history": messages
        })
        #prompts.append((item['id'], text, messages))
    
    # save first 5 prompts to txt file
    os.makedirs(args.out_dir + "/prompts", exist_ok=True)
    n_prompts_to_stamp = 5 
    logger.info(f"Saving {n_prompts_to_stamp} prompts to {args.out_dir}/prompts/example_prompts_{MODEL_NAME}.txt")
    with open(args.out_dir + f'/prompts/example_prompts_{MODEL_NAME}.txt', 'a') as f:
        for pr in prompts[:n_prompts_to_stamp]:
            f.write(f"ID: {pr['id']}\n")
            f.write(pr['prompt'])
            f.write("*"*100+'\n')

    logger.info(f"Prompt: {prompts[0]['prompt']}")
    
    if args.n_sampling > 1 and args.mode == "agent":
        import copy
        batches = [[copy.deepcopy(el) for _ in range(args.n_sampling)] for el in prompts]
    else:
        batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]


    logger.info(f"Number of prompts: {len(prompts) * args.n_out_sequences}")
    logger.info(f"Number of batches: {len(batches)}")
    logger.info(f"Number of prompts in each batch: {len(batches[0]) * args.n_out_sequences}")

    n_samples = args.n_out_sequences if args.mode == "cot" else args.n_sampling
    out_path = args.out_dir + f"/completions/{MODEL_NAME}/{args.mode}/n{n_samples}/{now_dir}"
    os.makedirs(out_path, exist_ok=True)

    start_time_all = time.time()
    for id_batch, batch in enumerate(tqdm(batches)):
        start_time_batch = time.time()  # Record start time
        if args.mode == "cot":

            ids = [el['id'] for el in batch]
            input_prompts = [el['prompt'] for el in batch]
            packages = [el['package'] for el in batch]

            outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

            for id_out, out in enumerate(outputs):
                
                completions = [o.text.strip() for o in out.outputs]

                for completion in completions:
                    if not "python" in args.dataset_path:
                        java_codes = []
                        if "OlympicCoder" in MODEL_NAME and "</think>" in completion:
                            completion = completion.split("</think>")[1].strip()
                        
                        java_codes = extract_java_code(completion)
                        java_codes = [packages[id_out] + "\n\n" + code for code in java_codes]
                
                        java_codes = [{'filename': extract_filename(java_code), "content": java_code} for java_code in java_codes]

                        with open(f"{out_path}/completions_{args.mode}.jsonl", 'a') as f:
                            json.dump({"id": ids[id_out], "code": java_codes, "completion": completion}, f, ensure_ascii=False)
                            f.write('\n')
                    else:
                        python_code = extract_python_code(completion)
                        res_dict = {"id": ids[id_out], "code": python_code, "completion": completion}
                        with open(f"{out_path}/completions_{args.mode}_python.jsonl", 'a') as f:
                            json.dump(res_dict, f, ensure_ascii=False)
                            f.write('\n')
                            
                        year = ids[id_out].split("_")[0].replace("oop","").strip()
                        session = ids[id_out].split("_")[1].strip()
                        results = exec_python_tests_and_parse(args, dataset, python_code=python_code, year=year, session=session, now_dir=now_dir, k=id_out if args.n_out_sequences > 1 else None)
                        logger.info(results)
                        result_json = {
                            "year": year,
                            "session": session,
                            "compile_errors": [],
                            "runtime_errors": results['runtime_errors'] if results['runtime_errors'] else [{}],
                            "compilation_passed": True,
                            "runtime_passed": True if results['overall_status'] == "OK" else False,
                            "test_details": results['details']
                        }
                        result_json = check_mandatory_tests(result_json, optional_conditions)
                        
                        with open(f"{out_path}/unittest_{args.mode}_python.jsonl", 'a') as f:
                            json.dump(result_json, f, ensure_ascii=False)
                            f.write('\n')

        elif args.mode == "agent":
            
            

            batch_data = [batch,[],[],[]]

            num_attempts = args.n_sampling if args.n_sampling > 1 else 1
            
            for n_round in range(args.n_rounds+1):
                
                if not batch_data[n_round]:
                    break

                logger.info(f"ROUND {n_round}")
                ids = [el['id'] for el in batch_data[n_round]]
                years = [el['id'].split("_")[0].replace("oop","").strip() for el in batch_data[n_round]]
                sessions = [el['id'].split("_")[1].strip() for el in batch_data[n_round]]
                packages = [el['package'] for el in batch_data[n_round]]
                input_prompts = [el['prompt'] for el in batch_data[n_round]]
                messages = [el['chat_history'] for el in batch_data[n_round]]

                #print("PROMPTS:", input_prompts)
                outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
                for id_out, out in enumerate(outputs):
                    completion = out.outputs[0].text
                    logger.info(f"N ROUND: {n_round}, ID OUT: {id_out}")
                    logger.info(f" ########### COMPLETION ############")

                    logger.info(f"{completion}")

                    messages[id_out].append({"role": "assistant", "content": completion.strip()})

                    
                    if "OlympicCoder" in MODEL_NAME and "</think>" in completion:
                        completion = completion.split("</think>")[1].strip()
                    
                    if "python" not in args.dataset_path:
                        code_language = "java"
                        java_codes = []
                        java_codes = extract_java_code(completion)
                        java_codes = [packages[id_out] + "\n\n" + code for code in java_codes]

                        #logger.info(f" ########### JAVA CODE ############")
                        #logger.info(f"{java_codes}")
                        exam_passed = False

                        if java_codes:
                            compile_errors, runtime_errors, test_details = exec_java_code(java_codes, years[id_out], sessions[id_out], now_dir, k=id_out if num_attempts > 1 else None)

                            result_json = {
                                "year": years[id_out],
                                "session": sessions[id_out],
                                "compile_errors": compile_errors,
                                "runtime_errors": runtime_errors,
                                "compilation_passed": False if compile_errors else True,
                                "runtime_passed": False if runtime_errors or compile_errors else True,
                                "test_details": test_details,
                                "chat_history": messages[id_out]
                            }

                            result_json = check_mandatory_tests(result_json, optional_conditions)

                        else:
                            # No Java code found - record this as an error
                            compile_errors = [{"year": years[id_out], "session": sessions[id_out], "error": "No valid Java code found in completion"}]  
                            result_json = {
                                "year": years[id_out],
                                "session": sessions[id_out],
                                "compile_errors": compile_errors,
                                "runtime_errors": [],
                                "compilation_passed": False,
                                "runtime_passed": False,
                                "test_details": "",
                                "chat_history": messages[id_out],
                                "runtime_passed_mandatory": False
                            }
                    else:
                        code_language = "python"
                        exam_passed = False
                        python_code = extract_python_code(completion)
                        # res_dict = {"id": ids[id_out], "code": python_code, "completion": completion}
                        # with open(f"{out_path}/completions_{args.mode}_python.jsonl", 'a') as f:
                        #     json.dump(res_dict, f, ensure_ascii=False)
                        #     f.write('\n')
                        
                        year = ids[id_out].split("_")[0].replace("oop","").strip()
                        session = ids[id_out].split("_")[1].strip()
                        results = exec_python_tests_and_parse(args, dataset, python_code=python_code, year=year, session=session, now_dir=now_dir, k=id_out if args.n_out_sequences > 1 else None)
                        logger.info(results)
                        compile_errors = results['compile_errors']
                        runtime_errors = results['runtime_errors']
                        result_json = {
                            "year": year,
                            "session": session,
                            "compile_errors": [],
                            "runtime_errors": results['runtime_errors'] if results['runtime_errors'] else [{}],
                            "compilation_passed": True,
                            "runtime_passed": True if results['overall_status'] == "OK" else False,
                            "test_details": results['details'],
                            "chat_history": messages[id_out]
                        }
                        result_json = check_mandatory_tests(result_json, optional_conditions)

                    if compile_errors:
                        
                        if "python" not in args.dataset_path:
                            messages[id_out].append({"role": "user", "content": f"Correct the compilation error. Rewrite your code from scratch while ensuring correctness.\n\n```output\n{compile_errors}\n```\n"})
                        else:
                            messages[id_out].append({"role": "user", "content": f"Correct the error. Modify only the necessary sections while preserving the rest of your code. Ensure that your response includes your full corrected code.\n\n```output\n{runtime_errors}\n```"})
                    
                    elif runtime_errors:
                        
                        if "python" not in args.dataset_path:
                            messages[id_out].append({"role": "user", "content": f"Correct the runtime error. Modify only the necessary sections while preserving the rest of your code. Ensure that your response includes your full corrected code.\n\n```output\n{runtime_errors}\n```"})
                        else:
                            messages[id_out].append({"role": "user", "content": f"Correct the runtime error. Modify only the necessary sections while preserving the rest of your code. Ensure that your response includes your full corrected code.\n\n```output\n{runtime_errors}\n```"})
                    else:
                        exam_passed = True
                        logger.info("EXAM PASSED!")

                        
                        messages[id_out].append({"role": "user", "content": f"Congrats, all tests passed!"})

                        with open(f"{out_path}/completions_{args.mode}_{code_language}.jsonl", 'a') as f:
                            json.dump(result_json, f, ensure_ascii=False)
                            f.write('\n')

                    if not exam_passed and n_round < args.n_rounds and messages[id_out]:
                        
                        text = tokenizer.apply_chat_template(
                            messages[id_out],
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        num_tokens = len(tokenizer.encode(text))
                        if num_tokens > args.max_model_len:
                            logger.info(f"Prompt too long ({num_tokens} tokens). Exam '{years[id_out]}_{sessions[id_out]}' not passed. Id attempt: {id_out}.")
                            
                            messages[id_out].append({"role": "user", "content": f"Reached maximum number of tokens. Exam not passed."})
                            
                            with open(f"{out_path}/completions_{args.mode}_{code_language}.jsonl", 'a') as f:
                                json.dump(result_json, f, ensure_ascii=False)
                                f.write('\n')

                            continue
                           
                        batch_data[n_round+1].append({
                            "id": ids[id_out],
                            "prompt": text,
                            "package": packages[id_out],
                            "chat_history": messages[id_out]}
                        )
                
                    if n_round == args.n_rounds and not exam_passed: # eached max possible rounds
                        logger.info("Exam NOT passed.")
                        #messages[id_out].append({"role": "assistant", "content": completion})
                                 
                        with open(f"{out_path}/completions_{args.mode}_{code_language}.jsonl", 'a') as f:
                            json.dump(result_json, f, ensure_ascii=False)
                            f.write('\n')
        
        end_time_batch = time.time()  # Record end time
        execution_time_batch = end_time_batch - start_time_batch

        logger.info(f"Execution Time Batch {id_batch}: {execution_time_batch:.4f} seconds")
    
    end_time_all = time.time()  # Record end time
    execution_time_all = end_time_all - start_time_all
    logger.info("----------------------------------")
    logger.info(f"Total Execution Time: {execution_time_all:.4f} seconds")