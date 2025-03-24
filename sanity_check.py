import os
import subprocess
import json
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compile and test Java exam sessions with JUnit.")
    parser.add_argument("--junit-jar", default= "lib/junit-platform-console-standalone-1.12.1.jar", required=True, help="Path to the JUnit standalone JAR file.")
    parser.add_argument("--exams-dir", default="exams", required=True, help="Base directory containing year-wise exam folders.")
    return parser.parse_args()

def main():
    args = parse_args()

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
        year_exams_dir = os.path.join(EXAMS_DIR, f"oop{year}-esami")
        
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
