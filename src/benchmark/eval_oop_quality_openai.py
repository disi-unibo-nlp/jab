import argparse
from dotenv import load_dotenv
import os
from datetime import datetime
import json

import logging
from tqdm import tqdm
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


# out/completions/Qwen2.5-Coder-32B-Instruct-AWQ/cot/n1/full_prec/completions_cot.jsonl
# out/completions/o4-mini-2025-04-16/cot/pass1/2025-04-18_13-46-52/completions_cot.jsonl
# out/completions/gpt-4.1-2025-04-14/cot/pass1/2025-04-21_21-36-59/completions_cot.jsonl
# out/completions/gemini-2.5-pro-exp-03-25/cot/pass1/2025-04-15_20-21-11/completions_cot.jsonl
# out/completions/deepseek-reasoner/cot/pass1/2025-04-18_10-33-58/completions_cot.jsonl
# gemini-2.5-flash-preview-04-17
def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark Gemini 2.0")
    parser.add_argument("--model_path", type=str, default="o4-mini-2025-04-16",
                        help="Path to the model to be used for benchmarking.")
    parser.add_argument("--completions_path", type=str, default="out/completions/o4-mini-2025-04-16/cot/pass1/2025-04-18_13-46-52/completions_cot.jsonl",
                        help="Path to the LLM completions.")
    parser.add_argument("--out_dir", type=str, default="out", 
                        help="Output directory for the benchmark results.")
    #parser.add_argument("--junit_path", type=str, default="out/completions/gpt-4.1-2025-04-14/cot/pass1/2025-04-21_21-36-59/junit_results.jsonl",
    #                    help="Path to the junit results.")
    
    return parser.parse_args()

def make_completion_openai_reasoner(sys_instr, prompt, model_name="o4-mini-2025-04-16", reasoning_effort="high"):

    # return response
    response = client.responses.create(
        model=model_name,
        reasoning={"effort": reasoning_effort},
        input=[
             {
                "role": "system", 
                "content": sys_instr
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )

    completion = response.output_text
    usage = response.usage
    usage_info = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens
    }

    return completion, usage_info

def make_completion_openai(system_instruction, prompt, model_name="gpt-4.1-2025-04-14"):
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system", 
                "content": system_instruction
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )

    completion = response.output_text
    usage = response.usage
    usage_info = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens
    }

    return completion, usage_info

def make_completion_deepseek(system_instruction, prompt, model_name):
    
    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model=model_name,
            messages = [
                {"role": "system","content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            seed=42
        )
        
        return response
    except Exception as e:
        print(e)
        return ""

if __name__ == "__main__":
    args = parse_arguments()
    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")

    if "deepseek" in args.model_path.lower():
        # Assert that OPENAI_API_KEY is set
        assert DEEPSEEK_API_KEY, "DEEPSEEK API KEY is required."
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
    elif "qwq" in args.model_path.lower():
        assert DASHSCOPE_API_KEY, "DASHSCOPE API KEY is required."
        client = OpenAI(
            api_key = DASHSCOPE_API_KEY,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
    else:
        assert OPENAI_API_KEY, "OPENAI API KEY is required."
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )
    
    OPENAI_MODEL_NAME =  args.model_path

    model_name = args.completions_path.split("/")[2]
    

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
You are an expert judge of Java code. You are given one or more Java code snippets that constitute a solution to an object-oriented programming (OOP) problem created by a student.
Your task is to evaluate the provided code according to the following OOP principles.
For each principle, provide a score on a scale of 1-3:

- 3 (Excellent) → The principle has been consistently and thoroughly adhered to across all relevant guidelines. The code exemplifies best practices and demonstrates a strong, consistent understanding of the principle, without significant violations.
- 2 (Satisfactory) → The principle has been adequately implemented, but there are minor inconsistencies or areas for improvement. The code demonstrates general understanding of the principle with some room for enhancement.
- 1 (Needs Improvement) → The principle has been poorly implemented or significantly violated. There are major issues that need to be addressed, showing limited understanding or application of the principle.

---

## 1. Code Clarity & Maintainability
**Goal:** The code should prioritize readability and straightforward logic, adhering to the KISS principle to ensure long-term maintainability.

**Guidelines:**
- Evaluate if variable and method names clearly communicate their purpose and functionality
- Field and parameters should be final as most as possible
- Check that methods are concise (generally under 20 lines) and perform a single, well-defined task
- Assess if control flow is straightforward, avoiding nested conditionals deeper than 2-3 levels
- Verify that comments explain "why" rather than "what" when logic isn't immediately obvious. Avoid to many comments
- Examine if code formatting is consistent and enhances readability

## 2. Object Design & Encapsulation
**Goal:** The solution should demonstrate proper encapsulation by protecting object state and exposing functionality through well-defined interfaces.

**Guidelines:**
- Verify that instance variables are appropriately private with controlled access through methods
- Check that classes expose clear, cohesive public interfaces that hide implementation details
- Check if the accessor modifiers are correctly used (also in costructors)
- Assess if immutability is used where appropriate to prevent unexpected state changes
- Evaluate if validation logic properly protects object invariants when state changes occur

## 3. Code Reuse & Modularity
**Goal:** The implementation should follow the DRY principle by eliminating duplication and organizing functionality into cohesive, loosely-coupled components.

**Guidelines:**
- Identify any repeated code blocks that could be extracted into reusable methods
- Assess if common behaviors are appropriately abstracted into shared methods
- Check that classes have high cohesion with focused, related responsibilities
- Evaluate if dependencies between classes are minimized to reduce coupling
- Verify that utility functions are placed in appropriate helper methods rather than duplicated

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
  "explanation": "A summary of the explanation of your evaluation, justifying the scores given.",
  "code_clarity_and_maintainability": <score_1_to_3>,
  "object_design_and_encapsulation": <score_1_to_3>,
  "code_reuse_and_modularity": <score_1_to_3>,
  "resource_management_and_efficiency": <score_1_to_3>
}
"""

    for i, item in enumerate(tqdm(dataset)):
        #completion = item['completion']
        codes = [el['filename'] + "\n\n" + el['content'] for el in item['code']]
        if codes:
            completion = "\n\n".join(codes).strip()
        else:
            completion = item['completion']
        
        prompt = f"Solution:\n{completion}"
        logger.info(f"Solution:\n{prompt}")

        if "o4" in args.model_path:
            completion, usage = make_completion_openai_reasoner(SYS_INSTRUCTION, prompt, OPENAI_MODEL_NAME)
        elif "gpt-4" in args.model_path:
            completion, usage = make_completion_openai(SYS_INSTRUCTION, prompt, OPENAI_MODEL_NAME)
        elif "deepseek" in args.model_path:
            response = make_completion_deepseek(SYS_INSTRUCTION, prompt, "deepseek-reasoner")
            completion = response.choices[0].message.content.strip()

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

        with open(f"{out_path}/quality_eval_binary_{OPENAI_MODEL_NAME}.jsonl", 'a') as f:
            res_dict = {"model_name": model_name, "id": item['id'], "scores": completion}
            if "o4" in args.model_path:
                res_dict["usage"] = usage
            json.dump(res_dict, f, ensure_ascii=False)
            f.write('\n')


        