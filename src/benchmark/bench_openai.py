import argparse
from dotenv import load_dotenv
import os
from datetime import datetime
import json
import time
import logging
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from src.benchmark.utils import extract_filename, extract_java_code, extract_and_remove_package_line, extract_python_code, create_exams, check_mandatory_tests, exec_python_tests_and_parse, exec_java_code
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument("--model_path", type=str, default="deepseek-reasoner", help="Model's HF directory or local path")
    parser.add_argument("--dataset_path", type=str, default="disi-unibo-nlp/JAB", help="Dataset HF directory")
    parser.add_argument("--out_dir", type=str, default="./out", help="Outputs directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of data to process in train set. Default is -1 to process all data.")
    parser.add_argument("--start_idx", type=int, default=0, help="Index of first prompt to process.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory to store model weights")
    parser.add_argument("--max_model_len", type=int, default=8000, help="Maximum input sequence length")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p sampling.")
    parser.add_argument("--n_samplings", type=int, default=1, help="Number of solutions to generate for a given prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature parameter")
    parser.add_argument("--mode", type=str, choices=["cot", "agent"], default='cot', help="Inference mode: CoT or Agent-based")
    parser.add_argument("--n_rounds", type=int, default=3, help="Number of rounds to use for inference.")
    parser.add_argument("--max_output_tokens", type=int, default=32768, help="Max number of tokens to generate in CoT prompting.")
    parser.add_argument("--year_sessions", type=str, default="", help="Specific years to consider, separated by comma. E.g: 2016,2018,2021")
    parser.add_argument("--junit_jar", default= "lib/junit-platform-console-standalone-1.13.0-M2.jar", help="Path to the JUnit standalone JAR file.")
    parser.add_argument("--logs_dir", default= "./logs", help="Path to the JUnit standalone JAR file.")
    parser.add_argument("--last_now_dir_path", default= "", help="To store results in the same path of last run.")
    parser.add_argument("--batch_api", action="store_true", default=False, help="Enable batch API mode if set.")

    return parser.parse_args()

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

def make_completion_openai_reasoner(prompt, model_name="o4-mini-2025-04-16", reasoning_effort="high"):

    # return response
    response = client.responses.create(
        model=model_name,
        reasoning={"effort": reasoning_effort},
        input=[
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

def make_completion_qwq(prompt):

    reasoning_content = ""  # Define complete thinking process
    answer_content = ""     # Define complete response
    is_answering = False   # Determine if thinking process has ended and response has begun
    usage = ""

    # Create chat completion request
    completion = client.chat.completions.create(
        model="qwq-plus", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        # QwQ model only supports streaming output calls
        stream=True,
        # Uncomment the following to return token usage in the last chunk
        stream_options={
            "include_usage": True
        }
    )

    #print("\n" + "=" * 20 + "Reasoning Process" + "=" * 20 + "\n")

    for chunk in completion:
        # If chunk.choices is empty, print usage
        if not chunk.choices:
            usage = chunk.usage
            #print("\nUsage:")
            #print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # Print thinking process
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                #print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # Start response
                if delta.content != "" and is_answering is False:
                    #print("\n" + "=" * 20 + "Complete Response" + "=" * 20 + "\n")
                    is_answering = True
                # Print response process
                #print(delta.content, end='', flush=True)
                answer_content += delta.content
    
    usage_info = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }

    return answer_content, reasoning_content, usage_info


def main():
   
    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")

    MODEL_NAME =  args.model_path
    
    dataset = load_dataset(args.dataset_path, split="test")
    #dataset = dataset.filter(lambda x: x['year'] == "2018" and x['session'] == "a03a")
    print(dataset)
    pass_k = args.n_samplings

    with open('data/optional_conditions.json') as f:
        optional_conditions = json.load(f)

    # Adjust dataset based on start_idx
    if args.start_idx > 0:
        assert args.start_idx < len(dataset), "start_idx is greater than the dataset size."
        dataset = dataset.select(range(args.start_idx, len(dataset)))

    # Adjust dataset based on max_samples
    if args.max_samples != -1:
        assert args.max_samples > 0, "max_samples should be greater than 0."
        dataset = dataset.select(range(args.max_samples))
    
    if args.mode == "agent" and "python" not in args.dataset_path:

        logger.info("Creating exams...")
        create_exams(args, dataset, now_dir)
        logger.info("Done!")

    logger.info(f"Processed {len(dataset)} records from {args.dataset_path}.")

    SYS_INSTRUCTION = """You are an expert Java developer. Your task is to solve the given university-level Java exam in a single attempt, ensuring that your solution correctly passes all JUnit tests defined in the provided `Test.java` file.  

You may use the provided utility Java files as needed. Your final answer must consist of one or more Java scripts, each enclosed separately between ```java and ``` in different code blocks."""

    if "python" in args.dataset_path:
        SYS_INSTRUCTION = """You are an expert Python developer. Your task is to solve the given university-level Python exam in a single attempt, ensuring that your solution correctly passes all unit tests defined in the provided `test.py` script.  

You may use the provided utilities as needed. Your final answer must consist of one Python script including only the solution to the problem, enclosed between ```python and ```."""


    out_path = args.out_dir + f"/completions/{MODEL_NAME}/{args.mode}/pass{pass_k}/{now_dir}"
    os.makedirs(out_path, exist_ok=True)
    total_prompts = 0
    for i, item in enumerate(tqdm(dataset)): 
        
        year = item['year']
        session = item['session']
        id_exam = f"oop{year}_{session}"


        if not "python" in args.dataset_path:
            PROMPT_TEMPLATE = """### Utility Files:\n\n{utility_files}\n\n### Test File:\n```java\n\n{test_file}```"""
            
            utilities = item['utility_classes']
            test_content = item['test']['content']

            package_line, test_content = extract_and_remove_package_line(java_code=test_content)
            test_file = f"// {item['test']['filename']}\n\n{test_content}"

            utility_files = ""
            for util_class in utilities:
                # fix consistency in package path
                _, util_class_content = extract_and_remove_package_line(util_class['content'].strip())
                utility_files += f"```java\n// {util_class['filename']}\n\n{util_class_content}```\n"

            prompt = PROMPT_TEMPLATE.format(utility_files=utility_files, test_file=test_file)

        else:
            PROMPT_TEMPLATE = """```python\n\n### Utility\n\n{utility}\n\n### Test\n\n{test}```"""
            utilities = item['utility_classes']
            test_content = item['test']['content']
            prompt = PROMPT_TEMPLATE.format(utility=utilities, test=test_content)


        if args.batch_api: 
            batch_request = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": args.model_path, "messages": [{"role": "system", "content": SYS_INSTRUCTION}], "temperature": args.temperature}}
            batch_request['body']["messages"].append({"role": "user", "content": prompt})
        
            
            for k in range(args.n_samplings):
                batch_request["custom_id"] = f"request-{id_exam}-{k}"
                with open(f'{out_path}/input_batch.jsonl', 'a') as f:
                    json.dump(batch_request, f, ensure_ascii=False)
                    f.write("\n")
                    total_prompts+=1
            
            continue

        for k in range(args.n_samplings):

            if args.mode == "cot":

                logger.info(f"EXAM: {item['year']}_{item['session']}")

                if "o4" in args.model_path.lower():
                    prompt = SYS_INSTRUCTION.replace("You are an expert Java developer.", "").strip() + "\n\n" + prompt
                    completion, usage = make_completion_openai_reasoner(prompt, MODEL_NAME)
                elif "gpt-4" in args.model_path.lower():
                    completion, usage = make_completion_openai(SYS_INSTRUCTION, prompt, MODEL_NAME)
                elif "deepseek" in args.model_path.lower():
                    response = make_completion_deepseek(SYS_INSTRUCTION, prompt, MODEL_NAME)
                    completion = response.choices[0].message.content.strip()
                elif "qwq" in args.model_path.lower():
                    prompt = SYS_INSTRUCTION.replace("You are an expert Java developer.", "").strip() + "\n\n" + prompt
                    completion, reasoning_content, usage = make_completion_qwq(prompt)
                    
                
                if args.model_path  == "deepseek-reasoner":
                    reasoning_content = response.choices[0].message.reasoning_content.strip()

                if not "python" in args.dataset_path:

                    java_codes = extract_java_code(completion)
                    java_codes = [package_line.strip() + "\n\n" + code for code in java_codes]
            
                    java_codes = [{'filename': extract_filename(java_code), "content": java_code} for java_code in java_codes]

                    with open(f"{out_path}/completions_{args.mode}.jsonl", 'a') as f:
                        res_dict = {"id": id_exam, "code": java_codes, "completion": completion}

                        if args.model_path  == "deepseek-reasoner" or "qwq" in args.model_path.lower():
                            res_dict["reasoning"] = reasoning_content

                        if "qwq" in args.model_path.lower() or "o4" in args.model_path.lower():
                            res_dict["usage"] = usage

                        json.dump(res_dict, f, ensure_ascii=False)
                        f.write('\n')

                    #time.sleep(5)

                else:
                    python_code = extract_python_code(completion)
                    logger.info(python_code)

                    results = exec_python_tests_and_parse(args, dataset, python_code=python_code, year=year, session=session, now_dir=now_dir, k=k if args.n_samplings > 1 else None)
                    logger.info(results)
                    result_json = {
                        "year": year,
                        "session": session,
                        "compile_errors": [],
                        "runtime_errors": results['runtime_errors'] if results['runtime_errors'] else [{}],
                        "compilation_passed": True,
                        "runtime_passed": True if results['overall_status'] == "OK" else False,
                        "test_details": results['details'],
                        
                    }
                    result_json = check_mandatory_tests(result_json, optional_conditions)
                
                    
                    with open(f"{out_path}/completions_{args.mode}_python.jsonl", 'a') as f:
                        json.dump({"id": id_exam, "code": python_code, "completion": completion}, f, ensure_ascii=False)
                        f.write('\n')
                    
                    with open(f"{out_path}/unittest_{args.mode}_python.jsonl", 'a') as f:
                        json.dump(result_json, f, ensure_ascii=False)
                        f.write('\n')

                    time.sleep(5)
            
            elif args.mode == "agent":
                pass
                # chat = client.chats.create(
                #     model=MODEL_NAME, 
                #     config=types.GenerateContentConfig(
                #     system_instruction=SYS_INSTRUCTION,
                #     temperature=args.temperature,
                #     #max_output_tokens=8192,
                #     top_p=args.top_p,)
                # )
                
                # exam_passed = False

                # for n_round in range(args.n_rounds+1):
                #     logger.info(f"EXAM {item['year']}_{item['session']}: ROUND {n_round+1}...")

                #     response = chat.send_message(prompt)
                #     completion = response.text

                #     chat_history = [] 
                #     for message in chat.get_history():
                #         chat_history.append({"role": message.role, "content": message.parts[0].text})

                #     if not "python" in args.dataset_path:
                #         java_codes = extract_java_code(completion)
                #         java_codes = [package_line.strip() + "\n\n" + code for code in java_codes]
                    
                #         if java_codes:
                #             compile_errors, runtime_errors, test_details = exec_java_code(args, logger, java_codes, year, session, now_dir, k if args.n_samplings > 1 else None)

                #             result_json = {
                #                 "year": year,
                #                 "session": session,
                #                 "compile_errors": compile_errors,
                #                 "runtime_errors": runtime_errors,
                #                 "compilation_passed": False if compile_errors else True,
                #                 "runtime_passed": False if runtime_errors or compile_errors else True,
                #                 "test_details": test_details,
                #                 "chat_history": chat_history
                #             }

                #             result_json = check_mandatory_tests(result_json, optional_conditions)

                            
                #         else:
                #             compile_errors = []
                #             # No Java code found - record this as an error
                #             compile_errors.append("Error: No valid Java code found. Ensure it is defined within ```java and ```.") 
                #             result_json = {
                #                 "year": year,
                #                 "session": session,
                #                 "compile_errors": compile_errors,
                #                 "runtime_errors": [],
                #                 "compilation_passed": False,
                #                 "runtime_passed": False,
                #                 "test_details": "",
                #                 "chat_history": chat_history,
                #                 "runtime_passed_mandatory": False
                #             }

                #         if compile_errors:
                #             prompt = f"Correct the compilation error. Rewrite your code from scratch while ensuring correctness.\n\n```output\n{compile_errors}\n```\n"
                        
                #         elif runtime_errors:
                #             prompt = f"Correct the runtime error. Modify only the necessary sections while preserving the rest of your code. Ensure that your response includes your full corrected code.\n\n```output\n{runtime_errors}\n```"
                        
                #         else:
                #             exam_passed = True
                #             logger.info("EXAM PASSED!")

                #             prompt = ""

                #             with open(f"{out_path}/completions_{args.mode}.jsonl", 'a') as f:
                #                 json.dump(result_json, f, ensure_ascii=False)
                #                 f.write('\n')

                #         if not exam_passed and n_round < args.n_rounds and chat_history:    
                #             continue
                    
                #         if n_round == args.n_rounds and not exam_passed: # eached max possible rounds
                            
                #             logger.info("Exam NOT passed.")
                            
                #             with open(f"{out_path}/completions_{args.mode}.jsonl", 'a') as f:
                #                 json.dump(result_json, f, ensure_ascii=False)
                #                 f.write('\n')
                        
                #         if exam_passed:
                #             break

                #         time.sleep(5)

                #     # run python exams
                #     else:
                #         python_code = extract_python_code(completion)
                #         logger.info(python_code)
                #         results = exec_python_tests_and_parse(args, dataset, python_code=python_code, year=year, session=session, now_dir=now_dir, k=k if args.n_samplings > 1 else None)
                #         logger.info(results)
                #         result_json = {
                #             "year": year,
                #             "session": session,
                #             "compile_errors": [],
                #             "runtime_errors": results['runtime_errors'] if results['runtime_errors'] else [{}],
                #             "compilation_passed": True,
                #             "runtime_passed": True if results['overall_status'] == "OK" else False,
                #             "test_details": results['details'],
                #             "chat_history": chat_history
                #         }
                #         result_json = check_mandatory_tests(result_json, optional_conditions)

                #         if results["overall_status"] != "OK":
                #             prompt = f"Correct the runtime error. Modify only the necessary sections while preserving the rest of your code. Ensure that your response includes your full corrected code.\n\n```output\n{results['runtime_errors']}\n```"
                #             logger.info(prompt)
                #         else: 
                #             exam_passed = True
                #             logger.info("EXAM PASSED!")

                #             prompt = ""

                #             with open(f"{out_path}/completions_{args.mode}_python.jsonl", 'a') as f:
                #                 json.dump(result_json, f, ensure_ascii=False)
                #                 f.write('\n')

                #         if not exam_passed and n_round < args.n_rounds and chat_history:    
                #             continue
                    
                #         if n_round == args.n_rounds and not exam_passed: # eached max possible rounds
                            
                #             logger.info("Exam NOT passed.")
                            
                #             with open(f"{out_path}/completions_{args.mode}_python.jsonl", 'a') as f:
                #                 json.dump(result_json, f, ensure_ascii=False)
                #                 f.write('\n')
                        
                #         if exam_passed:
                #             break

                #         time.sleep(5)

    ### Handling batch API
    if args.batch_api:
        logger.info(f"UNIQUE PROMPTS: {total_prompts / args.n_samplings}")
        logger.info(f"TOTAL PROMPTS: {total_prompts}")

        batch_input_file = client.files.create(
        file=open(f"{out_path}/input_batch.jsonl", "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch_obj = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "Running batch inference for JAB benchmark."
            }
        )
        logger.info(batch_obj)

        batch_id = batch_obj.id
        logger.info(f"BATCH ID: {batch_id}")

        with open(f'{out_path}/batch_id.txt', 'w') as f:
            f.write(batch_id)

                    

if __name__ == "__main__":
    args = parse_arguments()
    
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
    os.makedirs(args.logs_dir, exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"{args.logs_dir}/bench_{args.model_path}_{args.mode}.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(args)

    main()
    