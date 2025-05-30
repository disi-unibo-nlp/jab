import torch
import json
import os
import time
import argparse
import logging
import pandas as pd
import numpy as np 
import re
import subprocess

from tqdm import tqdm 
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types
from datasets import load_dataset
from src.benchmark.utils import extract_filename, extract_java_code, extract_and_remove_package_line, extract_python_code, exec_python_tests_and_parse

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Assert that GEMINI_API_KEY is set
assert GEMINI_API_KEY, "GEMINI API KEY is required."

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument("--model_path", type=str, default="gemini-2.0-flash-thinking-exp-01-21", help="Model's HF directory or local path")
    parser.add_argument("--dataset_path", type=str, default="disi-unibo-nlp/JAB", help="Dataset HF directory")
    parser.add_argument("--out_dir", type=str, default="./out", help="Outputs directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of data to process in train set. Default is -1 to process all data.")
    parser.add_argument("--start_idx", type=int, default=0, help="Index of first prompt to process.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory to store model weights")
    parser.add_argument("--max_model_len", type=int, default=8000, help="Maximum input sequence length")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p sampling.")
    parser.add_argument("--n_samplings", type=int, default=1, help="Number of solutions to generate for a given prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature parameter")
    parser.add_argument("--mode", type=str, choices=["cot", "agent"], default='cot', help="Inference mode: CoT or Agent-based")
    parser.add_argument("--n_rounds", type=int, default=3, help="Number of rounds to use for inference.")
    parser.add_argument("--max_output_tokens", type=int, default=32768, help="Max number of tokens to generate in CoT prompting.")
    parser.add_argument("--year_sessions", type=str, default="", help="Specific years to consider, separated by comma. E.g: 2016,2018,2021")
    parser.add_argument("--junit_jar", default= "lib/junit-platform-console-standalone-1.13.0-M2.jar", help="Path to the JUnit standalone JAR file.")
    parser.add_argument("--logs_dir", default= "./logs", help="Path to the JUnit standalone JAR file.")
    parser.add_argument("--last_now_dir_path", default= "", help="To store results in the same path of last run.")

    return parser.parse_args()

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

# def exec_python_tests_and_parse(dataset, python_code, year, session, now_dir, k=None, exams_dir="exams_python"):
#     year_dir = f"oop{year}"
#     session_dir = session
#     item_exam = dataset.filter(lambda x: str(x['year']) == str(year) and x['session'] == session)
   
#     python_code = item_exam['utility_classes'][0]['content'] + "\n\n" + python_code + "\n\n" + item_exam['test'][0]['content']

#     solution_dir = f"{args.out_dir}/{exams_dir}/{args.model_path.replace('.', '')}/{args.mode}/{now_dir}/{year_dir}/{session_dir}/sol1"
#     new_folder = f"k{k}" if not k is None else "pass1"
#     actual_sol_dir = solution_dir + f"/{new_folder}"

#     os.makedirs(actual_sol_dir, exist_ok=True)


    
#     code_filename = "test"
#     if code_filename and code_filename != "Test":
#         logger.info(f"Filename: {code_filename}")
#         #code = code.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;').replace('.sol1.',f'.sol1.{new_folder}.').replace('.sol1;', f'.sol1.{new_folder};')
#         with open(os.path.join(actual_sol_dir, code_filename + ".py"), 'w') as f:
#             f.write(python_code)

#     script_path = os.path.join(actual_sol_dir, code_filename).replace("./","").replace("/", ".")
#     command = ['python3', '-m', 'unittest', '-v', script_path]
#     process = subprocess.run(command, capture_output=True, text=True, check=False)

    
#     # After running the command and capturing output...
#     stdout_output = process.stdout
#     stderr_output = process.stderr
#     results = {
#         'runtime_errors': [],  # Initialize the key right away
#         'compile_errors': []   # Also initialize compile_errors
#     }
    
#     # Use all output for parsing
#     output_to_parse = stderr_output + stdout_output if stderr_output else stdout_output
    
#     fail_tests = []
#     num_tests = 0
#     found_tests = []
    
#     # First check for syntax errors
#     if "SyntaxError:" in output_to_parse:
#         results['overall_status'] = 'SYNTAX_ERROR'
#         compile_error = re.search(r'(SyntaxError:.+(\n.+)*)', output_to_parse)
#         compile_error_msg = compile_error.group(0) if compile_error else "Syntax error detected"
#         results['compile_errors'] = [compile_error_msg]
#         results['details'] = {
#             "tests_found": 0,
#             "tests_started": 0,
#             "tests_successful": 0,
#             "tests_failed": 0
#         }
#         return results
    
#     # Parse test results with more flexible regex patterns
#     test_patterns = [
#         # Pattern for test results with explicit status
#         re.compile(r'test_(\w+)\s+\(([\w.]+)\)\s+\.\.\.\s+(ok|ERROR|FAIL|SKIP)'),
#         # Pattern for test header lines in verbose output
#         re.compile(r'test_(\w+)\s+\(([\w.]+)\)')
#     ]
    
#     # Find all tests mentioned in the output
#     for line in output_to_parse.splitlines():
#         for pattern in test_patterns:
#             match = pattern.search(line)
#             if match:
#                 test_name = match.group(1)
#                 class_name = match.group(2)
#                 full_test_name = f"{class_name}.test_{test_name}"
                
#                 # Only count each test once
#                 if full_test_name not in found_tests:
#                     found_tests.append(full_test_name)
#                     num_tests += 1
                
#                 # If status is included in the pattern and it's not OK, add to fails
#                 if len(match.groups()) >= 3 and match.group(3).lower() != "ok":
#                     if f"test_{test_name}" not in fail_tests:
#                         fail_tests.append(f"test_{test_name}")
#                 break
    
#     # Also look for explicit error lines
#     error_pattern = re.compile(r'ERROR: test_(\w+)|FAIL: test_(\w+)')
#     for line in output_to_parse.splitlines():
#         error_match = error_pattern.search(line)
#         if error_match:
#             test_name = error_match.group(1) or error_match.group(2)
#             if f"test_{test_name}" not in fail_tests:
#                 fail_tests.append(f"test_{test_name}")
    
#     # If we haven't found any tests yet using patterns, try counting based on the summary
#     if num_tests == 0:
#         summary_pattern = re.compile(r'Ran (\d+) tests in')
#         summary_match = summary_pattern.search(output_to_parse)
#         if summary_match:
#             num_tests = int(summary_match.group(1))
    
#     # Determine overall status
#     overall_status_found = False
#     if "FAILED" in output_to_parse or "ERROR" in output_to_parse or fail_tests:
#         results['overall_status'] = 'FAIL'
#         overall_status_found = True
#     elif "OK" in output_to_parse and ("Ran" in output_to_parse or num_tests > 0):
#         results['overall_status'] = 'OK'
#         overall_status_found = True
    
#     if not overall_status_found:
#         results['overall_status'] = 'UNKNOWN'
    
#     # Fill in details
#     results['details'] = {
#         "tests_found": num_tests,
#         "tests_started": num_tests,
#         "tests_successful": num_tests - len(fail_tests),
#         "tests_failed": len(fail_tests)
#     }
    
#     # Include error information
#     if results['overall_status'] != 'OK':
#         results['runtime_errors'] = [{"fails": fail_tests, "error": output_to_parse}]
#     else:
#         results['runtime_errors'] = []
    
#     return results


def create_exams(dataset, now_dir, exams_dir="exams"):

    for item in tqdm(dataset, desc="Creating exams"):
        year = item['year']
        year_dir = f"oop{year}"
        session_dir = item['session']
        
        solution_dir = f"{args.out_dir}/{exams_dir}/{args.model_path}/{args.mode}/{now_dir}/{year_dir}/{session_dir}/sol1"

        range_to_consider = range(args.n_samplings)
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
                        .replace('.sol1.', f'.sol1.{new_folder}.')
                        .replace('.sol1;', f'.sol1.{new_folder};')
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
                )

                f.write(test_content)

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

def check_mandatory_tests(item, conditions):
    cond_key = f"oop{item['year']}_{item['session']}"
    
    if item["compilation_passed"] and not item["runtime_passed"]:
        if not "fails" in item["runtime_errors"][0]:
            # exception during previous execution
            item["runtime_passed_mandatory"] = False
            return item 

        fails = item["runtime_errors"][0]['fails']
        logger.info(f"Fail methods: {fails}")
        if not fails:
            item["runtime_passed_mandatory"] = True
            return item

        if not cond_key in conditions:
            if all('optional' in test.lower() for test in fails):
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False

        elif cond_key == "oop2018_a06":
            if ("testOne" not in fails and "testOneOf" not in fails and "testZeroOrMany" not in fails and "testSequence" not in fails) or ("testFullSequence" not in fails):
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False

        elif cond_key == "oop2022_a02a":
            if len(fails) == 1 and "testFromList" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        
        elif cond_key == "oop2022_a03a":
            if len(fails) == 1 and "testFinitePossibilities" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False

        elif cond_key == "oop2022_a02b": 
            if len(fails) == 1 and "testFromNonEmptyList" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        
        elif cond_key == "oop2022_a03b": 
            if len(fails) == 1 and "testConstant" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        
        elif cond_key == "oop2022_a04":
            if len(fails) == 1 and "testBasic" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        
        elif cond_key == "oop2021_a04":
            if len(fails) == 1 and fails[0] in "testFold/testFlatMap/testFilter":
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        elif cond_key == "oop2024_a01a":
            if len(fails) == 1 and fails[0] in "testOr/testSeq":
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        else:
            if len(fails) == 1:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False

    elif item["compilation_passed"] and item["runtime_passed"]:
        item["runtime_passed_mandatory"] = True
    else:
        item["runtime_passed_mandatory"] = False
    
    return item

def exec_java_code(java_files, year, session, now_dir, k=None, exams_dir="exams"):
    compile_errors = []
    runtime_errors = []

    year_dir = f"oop{year}"
    session_dir = session
    
    solution_dir = f"{args.out_dir}/{exams_dir}/{args.model_path}/{args.mode}/{now_dir}/{year_dir}/{session_dir}/sol1"
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


def main():
   
    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")

    MODEL_NAME =  args.model_path
    client = genai.Client(api_key=GEMINI_API_KEY)
    dataset = load_dataset(args.dataset_path, split="test")
    #dataset = dataset.filter(lambda x: x['year'] == "2018" and x['session'] == "a03a")
    print(dataset)
    pass_k = args.n_samplings

    with open('data/optional_conditions.json') as f:
        optional_conditions = json.load(f)

    # Adjust dataset based on start_idx
    if args.start_idx > 0:
        assert args.start_idx < len(dataset), "start_idx is greater than the dataset size."
        dataset = dataset.select(range(args.start_idx, len(dataset)))

    # Adjust dataset based on max_samples
    if args.max_samples != -1:
        assert args.max_samples > 0, "max_samples should be greater than 0."
        dataset = dataset.select(range(args.max_samples))
    
    if args.mode == "agent" and "python" not in args.dataset_path:

        logger.info("Creating exams...")
        create_exams(dataset, now_dir)
        logger.info("Done!")

    logger.info(f"Processed {len(dataset)} records from {args.dataset_path}.")

    SYS_INSTRUCTION = """You are an expert Java developer. Your task is to solve the given university-level Java exam in a single attempt, ensuring that your solution correctly passes all JUnit tests defined in the provided `Test.java` file.  

You may use the provided utility Java files as needed. Your final answer must consist of one or more Java scripts, each enclosed separately between ```java and ``` in different code blocks."""

    if "python" in args.dataset_path:
        SYS_INSTRUCTION = """You are an expert Python developer. Your task is to solve the given university-level Python exam in a single attempt, ensuring that your solution correctly passes all unit tests defined in the provided `test.py` script.  

You may use the provided utilities as needed. Your final answer must consist of one Python script including only the solution to the problem, enclosed between ```python and ```."""


    out_path = args.out_dir + f"/completions/{MODEL_NAME}/{args.mode}/pass{pass_k}/{now_dir}"
    os.makedirs(out_path, exist_ok=True)

    for i, item in enumerate(tqdm(dataset)): 
        
        year = item['year']
        session = item['session']
        id_exam = f"oop{year}_{session}"
        
        if not "python" in args.dataset_path:
            PROMPT_TEMPLATE = """### Utility Files:\n\n{utility_files}\n\n### Test File:\n```java\n\n{test_file}```"""
            
            utilities = item['utility_classes']
            test_content = item['test']['content']

            package_line, test_content = extract_and_remove_package_line(java_code=test_content)
            test_file = f"// {item['test']['filename']}\n\n{test_content}"

            utility_files = ""
            for util_class in utilities:
                # fix consistency in package path
                _, util_class_content = extract_and_remove_package_line(util_class['content'].strip())
                utility_files += f"```java\n// {util_class['filename']}\n\n{util_class_content}```\n"

            prompt = PROMPT_TEMPLATE.format(utility_files=utility_files, test_file=test_file)

        else:
            PROMPT_TEMPLATE = """```python\n\n### Utility\n\n{utility}\n\n### Test\n\n{test}```"""
            utilities = item['utility_classes']
            test_content = item['test']['content']
            prompt = PROMPT_TEMPLATE.format(utility=utilities, test=test_content)

        for k in range(args.n_samplings):

            if args.mode == "cot":
                logger.info(f"EXAM: {item['year']}_{item['session']}")
                response = ""

                if "gemini-2.5-flash" in MODEL_NAME:
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYS_INSTRUCTION,
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=24576
                                )
                            ),
                        )
                    prompt_token = response.usage_metadata.prompt_token_count
                    answer_tokens = response.usage_metadata.candidates_token_count
                    think_tokens = response.usage_metadata.thoughts_token_count
                    usage = {
                        "prompt_tokens": prompt_token,
                        "answer_tokens": answer_tokens,
                        "thinking_tokens": think_tokens
                    }
                else:
                    response = client.models.generate_content(
                        model=MODEL_NAME, 
                        contents=prompt, 
                        config=types.GenerateContentConfig(
                        system_instruction=SYS_INSTRUCTION,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        )
                    )

                completion = response.text

                if not "python" in args.dataset_path:

                    java_codes = extract_java_code(completion)
                    java_codes = [package_line.strip() + "\n\n" + code for code in java_codes]
            
                    java_codes = [{'filename': extract_filename(java_code), "content": java_code} for java_code in java_codes]

                    with open(f"{out_path}/completions_{args.mode}.jsonl", 'a') as f:
                        res_dict = {"id": id_exam, "code": java_codes, "completion": completion}
                        if "gemini-2.5-flash" in MODEL_NAME:
                            res_dict["usage"] = usage
                        json.dump(res_dict, f, ensure_ascii=False)
                        f.write('\n')

                    time.sleep(5)

                else:
                    python_code = extract_python_code(completion)
                    #logger.info(python_code)

                    results = exec_python_tests_and_parse(args, dataset, python_code=python_code, year=year, session=session, now_dir=now_dir, k=k if args.n_samplings > 1 else None)
                    logger.info(results)
                    result_json = {
                        "year": year,
                        "session": session,
                        "compile_errors": results['compile_errors'] if results['compile_errors'] else [],
                        "runtime_errors": results['runtime_errors'] if results['runtime_errors'] else [{}],
                        "compilation_passed": True if not results['compile_errors'] else False,
                        "runtime_passed": True if results['overall_status'] == "OK" else False,
                        "test_details": results['details']
                    }
                    result_json = check_mandatory_tests(result_json, optional_conditions)
                
                    
                    with open(f"{out_path}/completions_{args.mode}_python.jsonl", 'a') as f:
                        res_dict = {"id": id_exam, "code": python_code, "completion": completion}
                        if "gemini-2.5-flash" in MODEL_NAME:
                            res_dict["usage"] = usage
                        json.dump(res_dict, f, ensure_ascii=False)
                        f.write('\n')
                    
                    with open(f"{out_path}/unittest_{args.mode}_python.jsonl", 'a') as f:
                        json.dump(result_json, f, ensure_ascii=False)
                        f.write('\n')

                    time.sleep(5)
            
            elif args.mode == "agent":

                if "gemini-2.5-flash" in MODEL_NAME:
                    chat = client.chats.create(
                        model=MODEL_NAME, 
                        config=types.GenerateContentConfig(
                        system_instruction=SYS_INSTRUCTION,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=24576
                            )
                        ),
                    )
                else:
                    chat = client.chats.create(
                        model=MODEL_NAME, 
                        config=types.GenerateContentConfig(
                        system_instruction=SYS_INSTRUCTION,
                        temperature=args.temperature,
                        top_p=args.top_p,)
                    )
                
                exam_passed = False

                for n_round in range(args.n_rounds+1):
                    logger.info(f"EXAM {item['year']}_{item['session']}: ROUND {n_round+1}...")

                    response = chat.send_message(prompt)
                    completion = response.text

                    chat_history = [] 
                    for message in chat.get_history():
                        chat_history.append({"role": message.role, "content": message.parts[0].text})

                    if not "python" in args.dataset_path:
                        java_codes = extract_java_code(completion)
                        java_codes = [package_line.strip() + "\n\n" + code for code in java_codes]
                    
                        if java_codes:
                            compile_errors, runtime_errors, test_details = exec_java_code(java_codes, year, session, now_dir, k if args.n_samplings > 1 else None)

                            result_json = {
                                "year": year,
                                "session": session,
                                "compile_errors": compile_errors,
                                "runtime_errors": runtime_errors,
                                "compilation_passed": False if compile_errors else True,
                                "runtime_passed": False if runtime_errors or compile_errors else True,
                                "test_details": test_details,
                                "chat_history": chat_history
                            }

                            result_json = check_mandatory_tests(result_json, optional_conditions)

                            
                        else:
                            compile_errors = []
                            # No Java code found - record this as an error
                            compile_errors.append("Error: No valid Java code found. Ensure it is defined within ```java and ```.") 
                            result_json = {
                                "year": year,
                                "session": session,
                                "compile_errors": compile_errors,
                                "runtime_errors": [],
                                "compilation_passed": False,
                                "runtime_passed": False,
                                "test_details": "",
                                "chat_history": chat_history,
                                "runtime_passed_mandatory": False
                            }

                        if compile_errors:
                            prompt = f"Correct the compilation error. Rewrite your code from scratch while ensuring correctness.\n\n```output\n{compile_errors}\n```\n"
                        
                        elif runtime_errors:
                            prompt = f"Correct the runtime error. Modify only the necessary sections while preserving the rest of your code. Ensure that your response includes your full corrected code.\n\n```output\n{runtime_errors}\n```"
                        
                        else:
                            exam_passed = True
                            logger.info("EXAM PASSED!")

                            prompt = ""

                            with open(f"{out_path}/completions_{args.mode}.jsonl", 'a') as f:
                                json.dump(result_json, f, ensure_ascii=False)
                                f.write('\n')

                        if not exam_passed and n_round < args.n_rounds and chat_history:    
                            continue
                    
                        if n_round == args.n_rounds and not exam_passed: # eached max possible rounds
                            
                            logger.info("Exam NOT passed.")
                            
                            with open(f"{out_path}/completions_{args.mode}.jsonl", 'a') as f:
                                json.dump(result_json, f, ensure_ascii=False)
                                f.write('\n')
                        
                        if exam_passed:
                            break

                        time.sleep(5)

                    # run python exams
                    else:
                        python_code = extract_python_code(completion)
                        logger.info(python_code)
                        results = exec_python_tests_and_parse(args, dataset, python_code=python_code, year=year, session=session, now_dir=now_dir, k=k if args.n_samplings > 1 else None)

                        logger.info(results)
                        result_json = {
                            "year": year,
                            "session": session,
                            "compile_errors": [],
                            "runtime_errors": results['runtime_errors'] if results['runtime_errors'] else [{}],
                            "compilation_passed": True,
                            "runtime_passed": True if results['overall_status'] == "OK" else False,
                            "test_details": results['details'],
                            "chat_history": chat_history
                        }
                        result_json = check_mandatory_tests(result_json, optional_conditions)

                        if results["overall_status"] != "OK":
                            if results['compile_errors']:
                                prompt = f"The code was not able to run. Correct the error. Modify only the necessary sections while preserving the rest of your code. Ensure that your response includes your full corrected code.\n\n```output\n{results['compile_errors']}\n```"
                            else:
                                prompt = f"Correct the runtime error. Modify only the necessary sections while preserving the rest of your code. Ensure that your response includes your full corrected code.\n\n```output\n{results['runtime_errors']}\n```"
                            logger.info(prompt)
                        else: 
                            exam_passed = True
                            logger.info("EXAM PASSED!")

                            prompt = ""

                            with open(f"{out_path}/completions_{args.mode}_python.jsonl", 'a') as f:
                                json.dump(result_json, f, ensure_ascii=False)
                                f.write('\n')

                        if not exam_passed and n_round < args.n_rounds and chat_history:    
                            continue
                    
                        if n_round == args.n_rounds and not exam_passed: # eached max possible rounds
                            
                            logger.info("Exam NOT passed.")
                            
                            with open(f"{out_path}/completions_{args.mode}_python.jsonl", 'a') as f:
                                json.dump(result_json, f, ensure_ascii=False)
                                f.write('\n')
                        
                        if exam_passed:
                            break

                        time.sleep(5)


                    

if __name__ == "__main__":
    args = parse_arguments()
    
    os.makedirs(args.logs_dir, exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"{args.logs_dir}/bench_gemini_2.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(args)

    main()
    
    