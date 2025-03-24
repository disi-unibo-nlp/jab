import os
import re
import json
import pandas as pd
import argparse

from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Parse input data into JSONL file.")
    parser.add_argument("--exams_dir", default="exams-download", required=True, help="Base directory containing year-wise exam folders.")
    parser.add_argument("--output_path", default="./local_output/jab.jsonl", required=True, help="Path to Hugging Face data repository.")
    parser.add_argument("--exam_years", default="", help="List of specific exam year sessions to consider, splitted by comma. E.g: 2016,2018,2021")
    return parser.parse_args()

def remove_comments(code):
    """
    Removes comments from Java code.
    """
    # Remove single-line comments
    code = re.sub(r'//.*', '', code)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def split_java_file(java_code):
    """
    Splits a Java test file into its main components: imports, private variables, 
    initialization methods, and test functions.

    This function extracts:
    - Imports: All statements before the 'public class Test' declaration.
    - Private Variables and Initialization: Any private variables or @Before setup methods.
    - Test Methods: All methods annotated with @Test.

    Args:
        java_code (str): The full Java test file content as a string.

    Returns:
        tuple: A tuple containing:
            - imports (str): The import statements.
            - private_init (str): Private variables and initialization methods.
            - test_functions (list): A list of extracted @Test methods as formatted strings.
    """
    # Extract imports (everything before 'public class Test')
    imports_match = re.search(r'^(.*?)(?=public class Test)', java_code, re.DOTALL)
    imports = imports_match.group(1).strip() if imports_match else ""

    # Extract private variables and @Before initialization methods
    private_init_match = re.search(r'public class Test\s*\{(.*?)(?=\n\s*@org\.junit\.Test)', java_code, re.DOTALL)
    private_init = "\t" + private_init_match.group(1).strip() if private_init_match else ""

    # Extract all @Test methods
    test_functions = re.findall(r'(@org\.junit\.Test\s+public void (\w+)\(\) \{(.*?)\})', java_code, re.DOTALL)
    test_functions = [f"\t{full_match.strip()}" for full_match, _, _ in test_functions]

    return imports, private_init, test_functions

def parse_to_jsonl(exams_dir, year_range):
    """
    Parses exam data from multiple years and sessions into a structured JSONL format.

    This function iterates over a given range of years, extracting Java exam files,
    solutions, and test files from predefined directory structures. It removes comments 
    from exam code and organizes the extracted data into a list of dictionaries.

    Args:
        exams_dir (str): Base directory containing year-wise exam folders.
        year_range (iterable): List or range of years to process.

    Returns:
        list: A list of dictionaries, each containing the year, session, exam files,
              test file, and solution files.
    """
    out_year = []
    
    for year in year_range:
        root_path = f"{exams_dir}/oop{year}-esami"
        session_dirs = [el for el in os.listdir(root_path) if el.startswith('a0')]

        for session in session_dirs:
            sub_dirs = sorted([el for el in os.listdir(f'{root_path}/{session}') if el.endswith('1')])
            out_dict = {"exam": [], "test_ita": "", "solution": []}
            exam_files = []
            for sub_dir in sub_dirs:
                path = f'{root_path}/{session}/{sub_dir}'
                for filename in os.listdir(path):
                    if os.path.isfile(os.path.join(path, filename)) and filename.endswith(".java"):  # Ensures only files are considered
                        with open(os.path.join(path, filename), 'r') as f:
                            content = f.read().strip()
                            if sub_dir.startswith('sol'):
                                if filename.strip().lower().startswith('test'):
                                    out_dict['test_ita'] = {"filename": filename, "content": content}#, "components": {"imports": imports, "private_init": private_init, "test_functions": test_functions}}
                                elif filename.strip() not in exam_files:
                                    out_dict['solution'].append({"filename": filename, "content": content})

                            else:
                                
                                if not filename.strip().lower().startswith('test'):

                                    exam_files.append(filename.strip())
                                    content = remove_comments(content).strip()
                                    out_dict['exam'].append({"filename": filename, "content": content})
                            
            out_year.append({"year": year, "session": session, **out_dict})

    return out_year


def main():
    args = parse_args()
    range_to_consider = args.exam_years.split(",") if args.exam_years else range(2014, 2025)
    out_jsonl = parse_to_jsonl(exams_dir=args.exams_dir, year_range=range_to_consider)
    # Path to the file
    file_path = args.output_path
    ds = Dataset.from_pandas(pd.DataFrame(out_jsonl))
    print(ds)
    print(ds[0])
    # Check if the file already exists
    if not os.path.exists(file_path):
        # If the file does not exist, create and open it for appending
        with open(file_path, "a") as f:
            for item in out_jsonl:
                json.dump(item, f)
                f.write("\n")
    else:
        print(f"File {file_path} already exists. Skipping write operation.")

if __name__ == "__main__":
    main()