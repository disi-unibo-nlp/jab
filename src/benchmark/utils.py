import os
import pandas as pd
import numpy as np 
import subprocess
import re

def exec_java_code(java_codes, year, session, junit_jar, k=None, exams_dir="exams"):

    compile_errors = []
    exec_errors = []

    year_dir = f"oop{year}"
    session_dir = session
    
    solution_dir = f"{args.out_dir}/{exams_dir}/{MODEL_NAME}/{now_dir}/{year_dir}/{session_dir}/sol1"
    new_folder = f"k{k}" if not k is None else "pass1"
    actual_sol_dir = solution_dir + f"/{new_folder}"

    bin_path = os.path.join(actual_sol_dir, "bin")

    os.makedirs(bin_path, exist_ok=True)

    for root, _, files in os.walk(bin_path):
        for file in files:
            os.remove(os.path.join(root, file))
    
    for k, code in enumerate(java_codes):
        code_filename = extract_filename(code)
        if code_filename and code_filename != "Test":
            logger.info(f"Filename {k}: {code_filename}")
            code = code.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;').replace('.sol1;', f'.sol1.{new_folder};').replace('.sol1.Evaluation',f'.sol1.{new_folder}.Evaluation')
            with open(os.path.join(actual_sol_dir, code_filename + ".java"), 'w') as f:
                f.write(code)
        else:
            logger.info(f"This code is not a file to consider:\n\n{code}")

    java_files = sorted([os.path.join(actual_sol_dir, f) for f in os.listdir(actual_sol_dir) if f.endswith(".java")])
    
    if not java_files:
        logger.warning(f"No Java files found in {actual_sol_dir}. Skipping.")
        return

    logger.info(f"Compiling {len(java_files)} files in {actual_sol_dir}...")
    compile_command = ["javac", "-cp", junit_jar, "-d", bin_path] + java_files

    try:
        subprocess.run(compile_command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err_output = e.stderr.strip().replace("\n", " ")
        logger.error(f"Compilation failed for {actual_sol_dir}: {err_output}")
        compile_errors.append({"year": year, "session": session, "error": err_output})

    if not compile_errors:

        logger.info(f"Running tests for {actual_sol_dir}...")
        run_command = [
            "java", "-cp", f"{bin_path}:{junit_jar}",
            "org.junit.platform.console.ConsoleLauncher",
            "--class-path", bin_path,
            "--scan-class-path"
        ]

        try:
            result = subprocess.run(run_command, capture_output=True, text=True, timeout=10)
            logger.info(result.stdout)

            # If the command runs but tests fail, check stderr for failures
            if result.returncode != 0:
                logger.info(f"{actual_sol_dir}: Tests failed with errors.")
                
                exec_errors.append({
                    "year": year,
                    "session": session,
                    "error": "Failures " + result.stdout.split("Failures")[1].strip()
                })

        except subprocess.SubprocessError as e:
            logger.info(f"{actual_sol_dir}: failed with error {e}\n")
            exec_errors.append({
                {"year": year, "session": session, "error": result.stderr if 'result' in locals() else str(e)}
            })
        except TimeoutError:
            logger.info(f"{actual_sol_dir}: timed out after 10 seconds\n")
            exec_errors.append({
                {"year": year, "session": session, "error": "Timeout error."}
            })
    
    return compile_errors, exec_errors


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

def extract_and_remove_package_line(java_code):
    match = re.search(r'^package\s+[^\n]+;', java_code, flags=re.MULTILINE)
    package_line = match.group(0) if match else None
    cleaned_code = re.sub(r'^package\s+[^\n]+;\n?', '', java_code, flags=re.MULTILINE)
    return package_line, cleaned_code