import os
import subprocess
import glob
import json
import re
import logging
import argparse

from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument("--completions_path", type=str, default="out/completions/Qwen2.5-Coder-7B-Instruct/cot/pass10/2025-04-03_11-10-59/completions_cot.jsonl", help="Model's HF directory or local path")
    parser.add_argument("--model_names", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Dataset HF directory")
    parser.add_argument("--out_dir", type=str, default="./out", help="Outputs directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of data to process in train set. Default is -1 to process all data.")
    parser.add_argument("--start_idx", type=int, default=0, help="Index of first prompt to process.")
    parser.add_argument("--mode", type=str, choices=["cot", "tir"], default='cot', help="Inference mode: CoT or TIR")
    parser.add_argument("--max_tokens_tir", type=int, default=1024, help="Max number of tokens to generate in TIR prompting.")
    parser.add_argument("--year_sessions", type=str, default="", help="Specific years to consider, separated by comma. E.g: 2016,2018,2021")
    parser.add_argument("--junit_jar", default= "lib/junit-platform-console-standalone-1.13.0-M2.jar", help="Path to the JUnit standalone JAR file.")
    parser.add_argument("--k", type=int, default=10, help="value of K in Pass@k")

    return parser.parse_args()


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

def exec_java_code(java_files):
    compile_errors = []
    runtime_errors = []

    for root, _, files in os.walk(bin_path):
        for file in files:
            os.remove(os.path.join(root, file))

    logger.info(f"Compiling {len(java_files)} files in {actual_sol_dir}...")
    compile_command = ["javac", "-cp", JUNIT_JAR, "-d", bin_path] + java_files

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
                    "details": test_details,
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

    return compile_errors, runtime_errors

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

    else:
        item["runtime_passed_mandatory"] = False
    
    return item
            

if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="junit_test.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    args = parse_arguments()
    print(args)

    JUNIT_JAR = args.junit_jar

    dataset = load_dataset('disi-unibo-nlp/JAB', split="test")

    with open('data/optional_conditions.json') as f:
        optional_conditions = json.load(f)

    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))

    with open(args.completions_path) as f:
        completions = [json.loads(line) for line in f.readlines()]
    
    
    # Define the path
    model_names = args.model_names.split(",") #['DeepSeek-Coder-V2-Lite-Instruct'] #['CodeLlama-34b-Instruct-hf', "DeepSeek-Coder-V2-Lite-Instruct", "llama-3-70b-instruct-awq", "Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "starcoder2-15b-instruct-v0.1"]#["codegemma-1.1-7b-it", "CodeLlama-7b-Instruct-hf", "deepseek-coder-6.7b-instruct", "Meta-Llama-3.1-8B-Instruct", "Phi-3-mini-128k-instruct"] #Codestral-22B-v0.1"
    year_sessions = args.year_sessions.split(",") #["oop2014", "oop2015", "oop2016", "oop2017", "oop2018", "oop2019", "oop2020", "oop2021", "oop2022", "oop2023"] 
    now_dir = args.completions_path.split("/")[-2]

    if args.year_sessions:
        logger.info(f"Number of exam sessions before filtering: {len(completions)}")
        completions = [completion for completion in completions if completion['id'].split("_")[0].replace("oop","").strip() in year_sessions]
        logger.info(f"Number of exam sessions after filtering: {len(completions)}")

    # Group by 'id'
    grouped_data = defaultdict(list)
    for item in completions:
        grouped_data[item["id"]].append(item)

    # Convert to list of dictionaries
    grouped_list = [{"id": k, "attempts": v} for k, v in grouped_data.items()]

    res_dir = os.path.dirname(args.completions_path)
    res_file_path = res_dir + "/junit_results_2.jsonl"
    # Ensure the file is cleared at the start of execution
    open(res_file_path, "w").close() 
    
    for model_name in model_names:
        actual_model_name = model_name.split("/")[-1]
        for exam_session in grouped_list:
            
            attempts = exam_session['attempts']
            logger.info(f"Number of attempts for exam session {exam_session['id']}: {len(attempts)}")
            for k, attempt in enumerate(attempts):
                year = attempt['id'].split("_")[0].replace("oop","").strip()
                session = attempt['id'].split("_")[1].strip()
                year_dir = f"oop{year}"
            
                solution_dir = f"{args.out_dir}/exams/{actual_model_name}/{now_dir}/{year_dir}/{session}/sol1"

                new_folder = f"k{k}" if len(attempts) > 1 else "pass1"
                actual_sol_dir = solution_dir + f"/{new_folder}"

                logger.info(f"Saving Java files to: {actual_sol_dir}")
                os.makedirs(actual_sol_dir, exist_ok=True)

                lib_path = "lib/junit-platform-console-standalone-1.13.0-M2.jar"
                bin_path = os.path.join(project_root, actual_sol_dir, "bin")        

                # Ensure bin directory exists
                os.makedirs(bin_path, exist_ok=True)

                gold_example = dataset.filter(lambda x: x['year'] == year and x['session'] == session)[0]

                utility_files = gold_example['utility_classes']

                for utility in utility_files:
                    code_filename = utility['filename']
                    code_content = utility['content']

                    # Replace old suffixes with the new folder structure
                    code_content = (code_content
                        .replace(".e1;", ".sol1;")
                        .replace(".sol2;", ".sol1;")
                        .replace(".e2;", ".sol1;")
                        .replace('.sol1.', f'.sol1.{new_folder}.')
                        .replace(".sol1.Evaluation", f".sol1.{new_folder}.Evaluation")
                        .replace("import static ex2015.a01a.sol1.", f"import static ex2015.a01a.sol1.{new_folder}.")
                        .replace(".sol1;", f".sol1.{new_folder};")
                    )

                    with open(os.path.join(actual_sol_dir, code_filename), 'w') as f:
                        f.write(code_content)


                java_codes = attempt['code']

                for java_code in java_codes:
                    code_filename = java_code['filename']
                    code_content = java_code['content']
                    code_content = (code_content
                        .replace('.e1;','.sol1;')
                        .replace('.sol2;','.sol1;')
                        .replace('.e2;','.sol1;')
                        #.replace('.sol1.Evaluation',f'.sol1.{new_folder}.Evaluation')
                        .replace('.sol1.', f'.sol1.{new_folder}.')
                        .replace('.sol1;', f'.sol1.{new_folder};')
                    )
                    if code_filename:
                        with open(os.path.join(actual_sol_dir, code_filename + ".java"), 'w') as f:
                            f.write(code_content)
                
                # writing test 
                test_content = gold_example['test']['content']
                test_content = (
                    test_content.replace('.e1;', '.sol1;')
                                .replace('.sol2;', '.sol1;')
                                .replace('.e2;', '.sol1;')
                                .replace('.sol1.Evaluation', f'.sol1.{new_folder}.Evaluation')
                                .replace('.sol1.', f'.sol1.{new_folder}.')
                                .replace('.sol1;', f'.sol1.{new_folder};')
                )

                with open(os.path.join(actual_sol_dir, gold_example['test']['filename']), 'w') as f:
                    f.write(test_content)

                
                java_files = sorted([os.path.join(actual_sol_dir, f) for f in os.listdir(actual_sol_dir) if f.endswith(".java")])
            
                if not java_files:
                    logger.warning(f"No Java files found in {actual_sol_dir}. Skipping.")
                    continue
                
                compile_errors, runtime_errors = exec_java_code(java_files)

                result_json = {
                    "year": year,
                    "session": session,
                    "compile_errors": compile_errors,
                    "runtime_errors": runtime_errors,
                    "compilation_passed": False if compile_errors else True,
                    "runtime_passed": False if runtime_errors or compile_errors else True
                }

                result_json = check_mandatory_tests(result_json, optional_conditions)

                with open(res_file_path, "a") as f:
                    json.dump(result_json, f)
                    f.write("\n")

        logger.info(f"Junit results for model {actual_model_name} stored at {res_file_path}")