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
from datasets import load_dataset
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Assert that GEMINI_API_KEY is set
assert GEMINI_API_KEY, "GEMINI API KEY is required."

def parse_args():
    parser = argparse.ArgumentParser(description="Script for processing a dataset with Gemini.")

    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash-thinking-exp-01-21",
                        help="Model's HF directory or local path")
    parser.add_argument("--input_data_path", type=str, default="disi-unibo-nlp/JAB",
                        help="Path to input data from HF directory.")
    #parser.add_argument("--output_data_path", type=str, required=True,
    #                    help="Path to store data post processing.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum number of items to process in the dataset. Default is -1 to process all data.")
    parser.add_argument("--start_idx", type=int, default=7,
                        help="Index of first prompt to process.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top p sampling.")
    parser.add_argument("--n_attempts", type=int, default=5,
                        help="Number of attempts to get exact matching java code.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature parameter.")
    parser.add_argument("--out_fails_path", type=str, default="./fail_ids.txt",
                        help="Sampling temperature parameter.")
    parser.add_argument("--output_completion_dir", type=str, default="./completions/python_conversion",
                        help="Sampling temperature parameter.")
    parser.add_argument("--logs_dir", default="logs", 
                        help="Base directory containing logs.")


    return parser.parse_args()

import re

def java_to_python_filename(java_filename):
    # Remove the '.java' extension
    name_without_extension = java_filename[:-5]
    
    # Convert camelCase to snake_case
    snake_case_name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name_without_extension).lower()
    
    # Add the '.py' extension
    python_filename = f"{snake_case_name}.py"
    
    return python_filename

def extract_python_code(text):
    """Extracts Python code blocks enclosed between ```python and ```."""
    pattern = r"```python\s+([\s\S]*?)\s+```"
    matches = [match.strip() for match in re.findall(pattern, text)]
    return matches[0] if len(matches) > 0 else ""

def extract_and_remove_package_line(java_code):
    match = re.search(r'^package\s+[^\n]+;', java_code, flags=re.MULTILINE)
    package_line = match.group(0) if match else None
    cleaned_code = re.sub(r'^package\s+[^\n]+;\n?', '', java_code, flags=re.MULTILINE)
    return package_line, cleaned_code


def main():
   
    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_completion_dir + "/" + now_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output dir set to: {output_dir}")

    MODEL_NAME =  args.model_name 

    client = genai.Client(api_key=GEMINI_API_KEY)
    # chat = client.chats.create(
    #     model=MODEL_NAME, 
    #     config=types.GenerateContentConfig(
    #     system_instruction="You are a helpful assistant.",
    #     temperature=args.temperature,
    #     max_output_tokens=8192,
    #     top_p=args.top_p,)
    # )

    dataset = load_dataset(args.input_data_path, split="test")
    dataset = dataset.filter(lambda x: x['year'] == "2023")

    # Adjust dataset based on start_idx
    if args.start_idx:
        assert args.start_idx < len(dataset), "start_idx is greater than the dataset size."
        dataset = dataset.select(range(args.start_idx, len(dataset)))

    # Adjust dataset based on max_samples
    if args.max_samples != -1:
        assert args.max_samples > 0, "max_samples should be greater than 0."
        dataset = dataset.select(range(args.max_samples))

    logger.info(f"Processed {len(dataset)} records from {args.input_data_path}.")

    #logger.info(f"First sample:\n{dataset[0]}")

    for i, item in enumerate(tqdm(dataset)): 
        
        # chat = client.chats.create(
        #     model=MODEL_NAME, 
        #     config=types.GenerateContentConfig(
        #     system_instruction="You are a helpful assistant.",
        #     temperature=args.temperature,
        #     max_output_tokens=8192,
        #     top_p=args.top_p,)
        # )

        out_dict = {'year': item['year'], 'session': item['session'], 'python_code': {}}
        utilities = item['utility_classes']
        utility_files = "```java\n\n"
        for util_class in utilities:
            # fix consistency in package path
            _, util_class_content = extract_and_remove_package_line(util_class['content'].strip())
            utility_files += f"`{util_class_content}\n\n"
        utility_files = utility_files.strip() + "```"

        solution = item['solution']

        solution_files = "```java\n\n"
        for sol in solution: 
            _, sol_content = extract_and_remove_package_line(sol['content'].strip())
            solution_files += f"{sol_content}\n\n"
        solution_files = solution_files.strip() + "```"

        test_file = f"```java\n\n{item['test']['content']}```"

        guidelines = """**Guidelines:**
1.  **Class Definition:** Use `class ClassName:` syntax.
2.  **Constructor:** Replace Java constructors with `def __init__(self, ...):`.
3.  **Methods & Naming:**
    * Convert methods to `def method_name(self, ...):`.
    * Follow **PEP 8:** Use `snake_case` for methods, functions, and variables (instead of Java's `camelCase`). Class names remain `CamelCase`.
4.  **Attributes:** Define and access instance attributes using `self.attribute_name`.
5.  **Getters/Setters:** Replace Java's `getX()`/`setX(value)` with Python properties:
    * Use `@property` for the getter method (`def attribute_name(self):`).
    * Use `@attribute_name.setter` for the setter method (`def attribute_name(self, value):`).
6.  **Visibility:** Python doesn't enforce `private`/`protected`. Use convention: prefix internal attributes/methods with a single underscore (`_internal_member`).
7.  **Type Annotations:** Use Python's type hints extensively for clarity (e.g., `name: str`, `count: int`, `items: list[str]`, `-> bool`). Import from `typing` (`Optional`, `List`, `Dict`, `Set`, etc.) as needed, although built-ins (`list`, `dict`, `set`) are preferred in type hints in modern Python (3.9+).
8.  **Collections:** Map Java collections to Python built-ins:
    * `List<T>` -> `list[T]`
    * `Map<K, V>` -> `dict[K, V]`
    * `Set<T>` -> `set[T]`
9.  **Java-Specific Constructs:**
    * `Optional<T>`: Return the value or `None`. Use type hint `Optional[T]` or the newer `T | None`.
    * Utility classes like `Pair<A, B>` are often unnecessary in Python so REMOVE THEM — you can usually replace them with a simple tuple, which is more concise and idiomatic for common use cases.
    * Static methods/fields: Use the `@staticmethod` decorator or define attributes at the class level.
10. **String Representation:** Implement `__str__(self)` (user-friendly string, like `toString()`) and/or `__repr__(self)` (unambiguous developer representation).
11. **Equality:** Implement `__eq__(self, other)` (like `equals()`) and `__hash__(self)` (like `hashCode()`, if the object is immutable and needs hashing).
12. **Interfaces/Abstract Classes:** Use Python's `abc` module (Abstract Base Classes) if direct equivalents are needed."""
        
        system_instruction = (
            "Convert the following Java academic exam (with gold solution) into an equivalent Python academic exam (with gold solution). "
            "Specifically, you must:\n"
            "1. Convert Java classes to idiomatic, object-oriented Python classes.\n"
            "2. Convert JUnit tests into equivalent Python `unittest` tests.\n\n"
            f"{guidelines}\n\n"
            "**Output format**: Include all code — utility files, solution, and tests — in a single file enclosed within triple backticks (```python ... ```).\n\n"
            "**Important**: Adapt the exam instructions from the JUnit test file as comments within the corresponding Python `unittest` class definition (again as comments). "
            "Clearly separate the three sections in the Python code using the following comment headers: "
            "`# Utility Files`, `# Solution`, and `# Test File`." 
    )

        logger.info(f"SYSTEM:\n\n{system_instruction}\n\n")

        PROMPT_TEMPLATE = """### Utility Files:\n\n{utility_files}\n\n### Solution:\n\n{solution_files}\n\n### Test File:\n\n{test_file}"""
        prompt = PROMPT_TEMPLATE.format(utility_files=utility_files, solution_files=solution_files, test_file=test_file)
        logger.info(f"USER:\n\n{prompt}")

        response = client.models.generate_content(
            model=MODEL_NAME, 
            contents=prompt, 
            config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=args.temperature,
            #max_output_tokens=4096,
            top_p=args.top_p,
            )
        )
        python_code = extract_python_code(response.text)
        out_dict['python_code'] = python_code

        logger.info(f"ASSISTANT:\n\n{response.text}")
        logger.info("-----------------------------------------")
        # prompt = f"Convert the following Java script(s) into an equivalent unique Python script. Return as answer only the converted script, enclosed within ```python and ```.\n\n" + "\n\n".join([f"```java\n{el['content']}```" for el in utility])
        # #filenames = [el['filename'] for el in utility]
        # #filenames = [java_to_python_filename(filename) for filename in filenames]
        # # logger.info(f"Python filenames: {filenames}")
        # logger.info(f"Prompt:\n {prompt}")
        # response = ""

        # response = chat.send_message(prompt)
        # logger.info(response.text)

        # python_code = extract_python_code(response.text)
        # out_dict['utility_classes']['filename'] = "utility.py"
        # out_dict['utility_classes']['content'] = python_code

        # time.sleep(5)
        # prompt = f"Convert the following Java script(s) into a unqiue equivalent Python script. Return as answer only the converted script, enclosed within ```python and ```.\n\n" + "\n\n".join([f"```java\n{el['content']}```" for el in solution])

        # response = chat.send_message(prompt)
        # logger.info(response.text)

        # python_code = extract_python_code(response.text)
        # out_dict['solution']['filename'] = "solution.py"
        # out_dict['solution']['content'] = python_code
    
        # time.sleep(5)
        # prompt = f"Convert the following junit test script into an equivalent Python test script for Python-style interfaces/classes. Return as answer only the converted script, enclosed within ```python and ```.\n\n```java\n{test['content']}" 
        # response = chat.send_message(prompt)
        # logger.info(response.text)

        # python_code = extract_python_code(response.text)
        # out_dict['test']['filename'] = "test.py"
        # out_dict['test']['content'] = python_code


        # for message in chat.get_history():
        #     if message.role == "model":
        #         logger.info(f'role - {message.role}: {message.parts[0].text}')
        #         logger.info("-------------------------")

        with open(output_dir + "/converted_exams_python.jsonl", "a") as f:
            json.dump(out_dict, f)
            f.write("\n")
        
        # delete chat
        #del chat


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
    
    