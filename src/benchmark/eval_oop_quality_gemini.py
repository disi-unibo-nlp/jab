import os
import datetime
import argparse
import logging
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from google import genai
from google.genai import types
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Assert that GEMINI_API_KEY is set
assert GEMINI_API_KEY, "GEMINI API KEY is required."


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark Gemini 2.0")
    parser.add_argument("--model_path", type=str, default="gemini-2.5-flash-preview-04-17",
                        help="Path to the model to be used for benchmarking.")
    parser.add_argument("--completions_path", type=str, default="out/completions/gemini-2.5-pro-exp-03-25/cot/pass1/2025-04-15_20-21-11/completions_cot.jsonl",
                        help="Path to the LLM completions.")
    parser.add_argument("--out_dir", type=str, default="out", 
                        help="Output directory for the benchmark results.")
    #parser.add_argument("--junit_path", type=str, default="out/completions/gpt-4.1-2025-04-14/cot/pass1/2025-04-21_21-36-59/junit_results.jsonl",
    #                    help="Path to the junit results.")
    
    return parser.parse_args()

def main():
   
    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    GEMINI_MODEL_NAME =  args.model_path

    model_name = args.completions_path.split("/")[2]
    client = genai.Client(api_key=GEMINI_API_KEY)

    out_path = args.out_dir + f"/oop_quality_eval/{model_name}/{now_dir}"
    os.makedirs(out_path, exist_ok=True)

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"{out_path}/ouput.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(args)


    with open(args.completions_path, 'r') as f:
        dataset = [json.loads(line) for line in f.readlines()]

    #with open(args.junit_path, 'r') as f:
    #    junit = [json.loads(line) for line in f.readlines()]
    
    #id_exams = [f"oop{el['year']}_{el['session']}" for el in junit if "2023" == el['year']]
    #logger.info("Id exams selected: " + str(id_exams))
    dataset = [el for el in dataset if "2023" in el['id']]
    #dataset = dataset.filter(lambda x: x['year'] == "2018" and x['session'] == "a03a")
    logger.info("Dataset size: " + str(len(dataset)))
    

    SYS_INSTRUCTION = """
You are an expert judge of Java code. You are given one or more Java code snippets that constitute a solution to an object-oriented programming (OOP) problem made by a student. Your task is to evaluate the provided code according to the following OOP principles. For each principle, determine whether it has been adhered to or not.

---

## 1. Code Clarity & Maintainability
**Goal:** The code should prioritize readability and straightforward logic, adhering to the KISS principle to ensure long-term maintainability.
 
**Guidelines:**
- Evaluate if variable and method names clearly communicate their purpose and functionality
- Check that methods are concise (generally under 20 lines) and perform a single, well-defined task
- Assess if control flow is straightforward, avoiding nested conditionals deeper than 2-3 levels
- Verify that comments explain "why" rather than "what" when logic isn't immediately obvious
- Examine if code formatting is consistent and enhances readability
 
## 2. Object Design & Encapsulation
**Goal:** The solution should demonstrate proper encapsulation by protecting object state and exposing functionality through well-defined interfaces.
 
**Guidelines:**
- Verify that instance variables are appropriately private with controlled access through methods
- Check that classes expose clear, cohesive public interfaces that hide implementation details
- Assess if immutability is used where appropriate to prevent unexpected state changes
- Evaluate if validation logic properly protects object invariants when state changes occur
- Examine if inheritance relationships follow the Liskov Substitution Principle
 
## 3. Code Reuse & Modularity
**Goal:** The implementation should follow the DRY principle by eliminating duplication and organizing functionality into cohesive, loosely-coupled components.
 
**Guidelines:**
- Identify any repeated code blocks that could be extracted into reusable methods
- Assess if common behaviors are appropriately abstracted into shared methods or parent classes
- Check that classes have high cohesion with focused, related responsibilities
- Evaluate if dependencies between classes are minimized to reduce coupling
- Verify that utility functions are placed in appropriate helper classes rather than duplicated
 
## 4. Resource Management & Efficiency
**Goal:** The solution should demonstrate effective resource handling following proper initialization and cleanup patterns to prevent leaks and ensure optimal performance.
 
**Guidelines:**
- Check if resources (files, connections, etc.) are properly closed in finally blocks or try-with-resources
- Verify appropriate exception handling that maintains program stability and provides useful context
- Assess if collection types are chosen appropriately for the required operations
- Evaluate if unnecessary object creation is avoided, particularly in loops or recursive calls
- Examine if the solution avoids premature optimization while addressing obvious performance concerns

---

Your final answer must be ONLY a JSON object in the following format:

{
  "explanation": "A summary of the explanation of your evaluation",
  "code_clarity_and_maintainability": true/false,
  "object_design_and_encapsulation": true/false,
  "code_reuse_and_modularity": true/false,
  "resource_management_and_efficiency": true/false
}
"""

   
    for i, item in enumerate(tqdm(dataset)):
        completion = item['completion']
        prompt = f"Solution:\n{completion}"
        logger.info(f"Solution:\n{prompt}")

        if "gemini-2.5-flash" in GEMINI_MODEL_NAME:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
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
            raise ValueError("Model not supported")

        completion = response.text
        json_start = completion.find("{")
        json_end = completion.rfind("}")
        
        if json_start != -1 and json_end != -1:
            try:
                completion = completion[json_start:json_end + 1]
                completion = json.loads(completion)
            except json.JSONDecodeError:
                logger.error(f"JSONDecodeError: {completion}")
                continue
        else:
            logger.error(f"Invalid JSON response: {completion}")
            continue
        

        with open(f"{out_path}/quality_eval.jsonl", 'a') as f:
            res_dict = {"model_name": model_name, "id": item['id'], "scores": completion}
            if "gemini-2.5-flash" in GEMINI_MODEL_NAME:
                res_dict["usage"] = usage
            json.dump(res_dict, f, ensure_ascii=False)
            f.write('\n')


if __name__ == "__main__":
    args = parse_arguments()
    
    main()
    