import os
import subprocess
import json
import logging
import argparse

from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Compile and test Java exam sessions with JUnit.")
    parser.add_argument("--junit_jar", default= "lib/junit-platform-console-standalone-1.12.1.jar", required=True, help="Path to the JUnit standalone JAR file.")
    parser.add_argument("--exams_dir", default="exams", required=True, help="Base directory containing year-wise exam folders.")
    parser.add_argument("--dataset_path", default="disi-unibo-nlp/JAB", required=True, help="Path to Hugging Face data repository.")
    return parser.parse_args()

def create_exams(dataset, exams_dir):

    for item in tqdm(dataset, desc="Creating exams"):
        year = item['year']
        year_dir = f"oop{year}"
        session_dir = item['session']
        exam_dir = f"{exams_dir}/{year_dir}/{session_dir}/e1"
        solution_dir = f"{exams_dir}/{year_dir}/{session_dir}/sol1"
        os.makedirs(exam_dir, exist_ok=True)
        os.makedirs(solution_dir, exist_ok=True)

        # create utils
        for util_class in item['utility_classes']:
            util_class_filename = util_class['filename']
            with open(f'{exam_dir}/{util_class_filename}', 'w') as f:
                f.write(util_class['content'].strip())
            with open(f'{solution_dir}/{util_class_filename}', 'w') as f:
                f.write(util_class['content'].strip().replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;'))

        # create test
        test_filename = item['test']['filename']
        with open(f'{exam_dir}/{test_filename}', 'w') as f:
            f.write(item['test']['content'].strip())
        with open(f'{solution_dir}/{test_filename}', 'w') as f:
            test_content = item['test']['content'].strip()

            # correct specific human errors of mispelling
            if year == "2020" and session_dir == "a05":
                test_content = test_content.replace("createRechargableBattery", "createRechargeableBattery")
                test_content = test_content.replace("createSecureAndRechargableBattery", "createSecureAndRechargeableBattery")

            f.write(test_content.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;'))
        
        # create solutoon
        for sol in item['solution']:
            solution_filename = sol['filename']
            sol_content = sol['content'].strip()

            # correct specific human errors of mispelling
            if year == "2020" and session_dir == "a05":
                sol_content = sol_content.replace("createRechargableBattery", "createRechargeableBattery")
                sol_content = sol_content.replace("createSecureAndRechargableBattery", "createSecureAndRechargeableBattery")

            with open(f'{solution_dir}/{solution_filename}', 'w') as f:
                f.write(sol_content.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;'))

def main():
    args = parse_args()

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    dataset = load_dataset(args.dataset_path, split="test")

    if os.path.exists(args.exams_dir):
        logger.info(f"exams_dir: '{args.exams_dir}' already exists. Ignoring exam data creation.")
    else:
        create_exams(dataset=dataset, exams_dir=args.exams_dir)

    # check exam sanity
    JUNIT_JAR = args.junit_jar
    EXAMS_DIR = args.exams_dir

    if not os.path.isfile(JUNIT_JAR):
        raise FileNotFoundError(f"JUnit JAR file not found: {JUNIT_JAR}")

    if not os.path.isdir(EXAMS_DIR):
        raise NotADirectoryError(f"Exam directory not found: {EXAMS_DIR}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    compile_errors = []
    safe_sessions = []

    for year in range(2014, 2025):
        year_exams_dir = os.path.join(EXAMS_DIR, f"oop{year}")
        
        if not os.path.isdir(year_exams_dir):
            logger.warning(f"Skipping {year_exams_dir}: Directory does not exist.")
            continue

        exam_sessions = [item for item in os.listdir(year_exams_dir) if item.startswith("a0")]

        for session in exam_sessions:

            logger.info(f"Processing YEAR: {year}, SESSION: {session}")
            bin_path = os.path.join(project_root, "bin")

            os.makedirs(bin_path, exist_ok=True)

            for root, _, files in os.walk(bin_path):
                for file in files:
                    os.remove(os.path.join(root, file))

            test_dir = os.path.join(year_exams_dir, session, "sol1")
            java_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".java")]
            
            if not java_files:
                logger.warning(f"No Java files found in {test_dir}. Skipping.")
                continue

            logger.info(f"Compiling {len(java_files)} files in {test_dir}...")
            compile_command = ["javac", "-cp", JUNIT_JAR, "-d", bin_path] + java_files

            try:
                subprocess.run(compile_command, check=True, text=True, capture_output=True)
                safe_sessions.append({"year": year, "session": session})
            except subprocess.CalledProcessError as e:
                err_output = e.stderr.strip().replace("\n", " ")
                logger.error(f"Compilation failed for {test_dir}: {err_output}")
                compile_errors.append({"year": year, "session": session, "err": err_output})
                continue

            logger.info(f"Running tests for {test_dir}...")
            run_command = [
                "java", "-cp", f"{bin_path}:{JUNIT_JAR}",
                "org.junit.platform.console.ConsoleLauncher",
                "--class-path", bin_path,
                "--scan-class-path"
            ]

            try:
                result = subprocess.run(run_command, capture_output=True, text=True, timeout=10)
                logger.info(result.stdout)
            except subprocess.TimeoutExpired:
                logger.error(f"Test execution timed out for {test_dir}. Skipping.")

    logger.info(f"Safe sessions: {len(safe_sessions)}")
    if compile_errors:
        with open("sanity_check_errors.jsonl", "a") as f:
            for error in compile_errors:
                json.dump(error, f)
                f.write("\n")
        logger.info(f"Compilation errors recorded: {len(compile_errors)}")
    else:
        logger.info("All compilations succeeded!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="sanity_check.log",
                        filemode='w')
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    main()
