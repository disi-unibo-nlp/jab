import torch
import json
import os
import time
import argparse
import logging
import pandas as pd
import numpy as np 
import re

from tqdm import tqdm 
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Assert that GEMINI_API_KEY is set
assert GEMINI_API_KEY, "GEMINI API KEY is required."

def parse_args():
    parser = argparse.ArgumentParser(description="Script for processing a dataset with Gemini.")

    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash-001",
                        help="Model's HF directory or local path")
    parser.add_argument("--input_data_path", type=str, required=True,
                        help="Path to input data.")
    parser.add_argument("--output_data_path", type=str, required=True,
                        help="Path to store data post processing.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum number of items to process in the dataset. Default is -1 to process all data.")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Index of first prompt to process.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top p sampling.")
    parser.add_argument("--n_attempts", type=int, default=5,
                        help="Number of attempts to get exact matching java code.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature parameter.")
    parser.add_argument("--out_fails_path", type=str, default="./fail_ids.txt",
                        help="Sampling temperature parameter.")
    parser.add_argument("--output_completion_dir", type=str, default="./completions/translation",
                        help="Sampling temperature parameter.")
    parser.add_argument("--logs_dir", default="logs", 
                        help="Base directory containing logs.")


    return parser.parse_args()

def extract_java_code(text):
    """Extracts Java code blocks enclosed between ```java and ```."""
    pattern = r"```java\s+([\s\S]*?)\s+```"
    matches = [match.strip() for match in re.findall(pattern, text)]
    return matches[0] if len(matches) > 0 else ""

def clean_code_for_comparison(code):
    """
    Removes comments and string literals from Java code for comparison purposes.
    """
    # Remove single-line comments
    code = re.sub(r'//.*', '', code)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove string literals (keeping empty quotes to avoid syntax issues)
    code = re.sub(r'".*?"', '""', code)
    
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code).strip()
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
    pattern = r'(@org\.junit\.Test\s+public void (\w+)\s*\(\)\s*(?:throws \w+\s*)?\{([\s\S]*?)\})'
    matches = re.findall(pattern, java_code)
    test_functions = [f"\t{full_match.strip()}" for full_match, _, _ in matches]


    return imports, private_init, test_functions


def compare_java_code(original, translated):
    """
    Compares two Java code snippets after removing comments and string literals.
    """
    clean_original = clean_code_for_comparison(original)
    clean_translated = clean_code_for_comparison(translated)
    return clean_original == clean_translated

def main():
   
    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_completion_dir + "/" + now_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output dir set to: {output_dir}")

    MODEL_NAME =  args.model_name 

    client = genai.Client(api_key=GEMINI_API_KEY)

    with open(args.input_data_path) as f:
        dataset = [json.loads(line) for line in f.readlines()]

    # Adjust dataset based on start_idx
    if args.start_idx:
        assert args.start_idx < len(dataset), "start_idx is greater than the dataset size."
        dataset = dataset[args.start_idx:]

    # Adjust dataset based on max_samples
    if args.max_samples != -1:
        assert args.max_samples > 0, "max_samples should be greater than 0."
        dataset = dataset[:args.max_samples]

    logger.info(f"Processed {len(dataset)} records from {args.input_data_path}.")

    logger.info(f"First sample:\n{dataset[0]}")

    for i, item in enumerate(tqdm(dataset)): 
        original_code = item['test_ita']['content']
        prompt = f"Simply copy and paste the provided code exactly as it is, but translate all instructions and comments from Italian to English.\n\nThis is the code:\n\n{original_code}"
      
        for k in range(args.n_attempts):
            logger.info(f"{item['year']}_{item['session']}: ATTEMPT {k+1}...")
            response = ""

            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=prompt, 
                config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
                temperature=args.temperature,
                max_output_tokens=4096,
                top_p=args.top_p,
                )
            )

            with open(f"{output_dir}/completions.jsonl", "a") as f:
                compl_dict = {"year": item['year'], "session": item['session'], "completion": response.text.strip()}
                json.dump(compl_dict, f)
                f.write("\n")

            java_code_en = extract_java_code(response.text) 
            is_the_same = compare_java_code(original=original_code, translated=java_code_en)

            if is_the_same:
                logger.info("CODE MATCH: ok!")
                imports, private_init, test_functions = split_java_file(java_code_en)
                item['test'] = {"filename": item['test_ita']['filename'], "content": java_code_en, "components": {"imports": imports, "private_init": private_init, "test_functions": test_functions}}
                
                with open(args.output_data_path, "a") as f:
                    json.dump(item, f)
                    f.write("\n")
                break
            else:
                logger.info("CODE MATCH: error!")
                time.sleep(5)

        time.sleep(5)
        
        if not is_the_same:
            with open(args.out_fails_path, "a") as f:
                f.write(str(i) + "\n")

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.logs_dir, exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"{args.logs_dir}/translate.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(args)

    main()
    
    