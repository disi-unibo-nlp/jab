import os
import pandas as pd
import numpy as np 
import subprocess
import re
from tqdm import tqdm




def extract_java_code(text):
    """Extracts Java code blocks enclosed between ```java and ```."""
    pattern = r"```java\s+([\s\S]*?)\s+```"
    matches = [match.strip() for match in re.findall(pattern, text)]
    return matches if len(matches) > 0 else ""

def extract_python_code(text):
    """Extracts python code blocks enclosed between ```python and ```."""
    pattern = r"```python\s+([\s\S]*?)\s+```"
    matches = [match.strip() for match in re.findall(pattern, text)]
    return matches[0] if len(matches) > 0 else ""


# def extract_filename(java_code):
#     # Matches class, interface, enum, annotation, and record declarations
#     pattern = r"(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?(?:class|interface|enum|@interface|record)\s+([A-Za-z_][A-Za-z0-9_]*)"

#     match = re.search(pattern, java_code)
#     return match.group(1) if match else None

def extract_filename(java_code):
    # Remove multi-line comments first
    code_without_comments = re.sub(r'/\*[\s\S]*?\*/', '', java_code)
    
    # Match class declarations with capital-letter class names
    pattern = r"(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?(?:class|interface|enum|@interface|record)\s+([A-Z][A-Za-z0-9_]*)"
    
    match = re.search(pattern, code_without_comments)
    return match.group(1) if match else None

def extract_and_remove_package_line(java_code):
    match = re.search(r'^package\s+[^\n]+;', java_code, flags=re.MULTILINE)
    package_line = match.group(0) if match else None
    cleaned_code = re.sub(r'^package\s+[^\n]+;\n?', '', java_code, flags=re.MULTILINE)
    return package_line, cleaned_code

def check_mandatory_tests(item, conditions):
    cond_key = f"oop{item['year']}_{item['session']}"
    
    if item["compilation_passed"] and not item["runtime_passed"]:
        if not "fails" in item["runtime_errors"][0]:
            # exception during previous execution
            item["runtime_passed_mandatory"] = False
            return item 

        fails = item["runtime_errors"][0]['fails']
        print(f"Fail methods: {fails}")
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

def create_exams(args, dataset, now_dir, exams_dir="exams"):

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

# def exec_python_tests_and_parse(args, dataset, python_code, year, session, now_dir, k=None, exams_dir="exams_python"):
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

def exec_python_tests_and_parse(args, dataset, python_code, year, session, now_dir, k=None, exams_dir="exams_python", timeout=30):
    
    year_dir = f"oop{year}"
    session_dir = session
    item_exam = dataset.filter(lambda x: str(x['year']) == str(year) and x['session'] == session)
   
    python_code = item_exam['utility_classes'][0]['content'] + "\n\n" + python_code + "\n\n" + item_exam['test'][0]['content']

    solution_dir = f"{args.out_dir}/{exams_dir}/{args.model_path.replace('.', '')}/{args.mode}/{now_dir}/{year_dir}/{session_dir}/sol1"
    new_folder = f"k{k}" if not k is None else "pass1"
    actual_sol_dir = solution_dir + f"/{new_folder}"

    os.makedirs(actual_sol_dir, exist_ok=True)
    
    code_filename = "test"
    if code_filename and code_filename != "Test":
        with open(os.path.join(actual_sol_dir, code_filename + ".py"), 'w') as f:
            f.write(python_code)

    script_path = os.path.join(actual_sol_dir, code_filename).replace("./","").replace("/", ".")
    command = ['python3', '-m', 'unittest', '-v', script_path]
    
    # Initialize results dictionary with default values
    results = {
        'runtime_errors': [],
        'compile_errors': [],
        'details': {
            "tests_found": 0,
            "tests_started": 0,
            "tests_successful": 0,
            "tests_failed": 0
        }
    }
    
    try:
        # Run with timeout
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
        
        # After running the command and capturing output...
        stdout_output = process.stdout
        stderr_output = process.stderr
        
        # Use all output for parsing
        output_to_parse = stderr_output + stdout_output if stderr_output else stdout_output
        
        # Existing parsing logic...
        fail_tests = []
        num_tests = 0
        found_tests = []
        
        # First check for syntax errors
        if "SyntaxError:" in output_to_parse:
            results['overall_status'] = 'SYNTAX_ERROR'
            compile_error = re.search(r'(SyntaxError:.+(\n.+)*)', output_to_parse)
            compile_error_msg = compile_error.group(0) if compile_error else "Syntax error detected"
            results['compile_errors'] = [compile_error_msg]
            results['details'] = {
                "tests_found": 0,
                "tests_started": 0,
                "tests_successful": 0,
                "tests_failed": 0
            }
            return results
        
        # Parse test results with more flexible regex patterns
        test_patterns = [
            # Pattern for test results with explicit status
            re.compile(r'test_(\w+)\s+\(([\w.]+)\)\s+\.\.\.\s+(ok|ERROR|FAIL|SKIP)'),
            # Pattern for test header lines in verbose output
            re.compile(r'test_(\w+)\s+\(([\w.]+)\)')
        ]
        
        # Find all tests mentioned in the output
        for line in output_to_parse.splitlines():
            for pattern in test_patterns:
                match = pattern.search(line)
                if match:
                    test_name = match.group(1)
                    class_name = match.group(2)
                    full_test_name = f"{class_name}.test_{test_name}"
                    
                    # Only count each test once
                    if full_test_name not in found_tests:
                        found_tests.append(full_test_name)
                        num_tests += 1
                    
                    # If status is included in the pattern and it's not OK, add to fails
                    if len(match.groups()) >= 3 and match.group(3).lower() != "ok":
                        if f"test_{test_name}" not in fail_tests:
                            fail_tests.append(f"test_{test_name}")
                    break
        
        # Also look for explicit error lines
        error_pattern = re.compile(r'ERROR: test_(\w+)|FAIL: test_(\w+)')
        for line in output_to_parse.splitlines():
            error_match = error_pattern.search(line)
            if error_match:
                test_name = error_match.group(1) or error_match.group(2)
                if f"test_{test_name}" not in fail_tests:
                    fail_tests.append(f"test_{test_name}")
        
        # If we haven't found any tests yet using patterns, try counting based on the summary
        if num_tests == 0:
            summary_pattern = re.compile(r'Ran (\d+) tests in')
            summary_match = summary_pattern.search(output_to_parse)
            if summary_match:
                num_tests = int(summary_match.group(1))
        
        # Determine overall status
        overall_status_found = False
        if "FAILED" in output_to_parse or "ERROR" in output_to_parse or fail_tests:
            results['overall_status'] = 'FAIL'
            overall_status_found = True
        elif "OK" in output_to_parse and ("Ran" in output_to_parse or num_tests > 0):
            results['overall_status'] = 'OK'
            overall_status_found = True
        
        if not overall_status_found:
            results['overall_status'] = 'UNKNOWN'
        
        # Fill in details
        results['details'] = {
            "tests_found": num_tests,
            "tests_started": num_tests,
            "tests_successful": num_tests - len(fail_tests),
            "tests_failed": len(fail_tests)
        }
        
        # Include error information
        if results['overall_status'] != 'OK':
            results['runtime_errors'] = [{"fails": fail_tests, "error": output_to_parse}]
        else:
            results['runtime_errors'] = []
    
    except subprocess.TimeoutExpired:
        # Handle timeout case
        results['overall_status'] = 'TIMEOUT'
        results['runtime_errors'] = [{
            "fails": ["all_tests_due_to_timeout"],
            "error": f"Test execution timed out after {timeout} seconds. This might indicate an infinite loop or a blocking operation in the code."
        }]
    
    except Exception as e:
        # Handle any other exceptions that might occur
        results['overall_status'] = 'ERROR'
        results['runtime_errors'] = [{
            "fails": ["execution_error"],
            "error": f"An error occurred during test execution: {str(e)}"
        }]
    
    return results

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

def exec_java_code(args, logger, java_files, year, session, now_dir, k=None, exams_dir="exams"):
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
