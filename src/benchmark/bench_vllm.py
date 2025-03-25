import torch
import json
import os
import logging
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
import re
from datetime import datetime

# Load variables from the .env file
load_dotenv()

# Manually set the required environment variable
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="open-r1/OlympicCoder-7B", metadata={"help": "model's HF directory or local path"})
    dataset_path: Optional[str] = field(default="disi-unibo-nlp/JAB",  metadata={"help": "dataset HF directory"})
    out_dir: Optional[str] =  field(default="./out", metadata={"help": "outputs directory"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    batch_size: Optional[int] = field(default=16, metadata={"help": "Maximum number of data to process per batch."})
    cache_dir: Optional[str] =  field(default=None, metadata={"help": "cache dir to store model weights"})
    max_model_len: Optional[int] = field(default=8000, metadata={"help": "Maximum input sequence length"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    n_sampling: Optional[int] = field(default=1, metadata={"help": "Number of solutions to generate for a given prompt"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default='cot', metadata={"help": "Inference mode: CoT or TIR", "choices":["cot", "tir"]})
    text_only: Optional[bool] = field(default=True, metadata={"help": 'whether to consider only textual question without images.'})
    n_gpus: Optional[int] = field(default=1, metadata={"help": "Number of gpus to use for inference."})
    n_rounds: Optional[int] = field(default=3, metadata={"help": "Number of gpus to use for inference."})
    max_tokens_cot: Optional[int] = field(default=32768, metadata={"help": "max number of tokens to generate in CoT prompting."})
    max_tokens_tir: Optional[int] = field(default=1024, metadata={"help": "max number of tokens to generate in TIR prompting."})
    year_sessions: Optional[str] = field(default="", metadata={"help": "specific years to consider, separated by comma. E.g: 2016,2018,2021"})

def extract_and_remove_package_line(java_code):
    match = re.search(r'^package\s+[^\n]+;', java_code, flags=re.MULTILINE)
    package_line = match.group(0) if match else None
    cleaned_code = re.sub(r'^package\s+[^\n]+;\n?', '', java_code, flags=re.MULTILINE)
    return package_line, cleaned_code

def extract_java_code(text):
    """Extracts Java code blocks enclosed between ```java and ```."""
    pattern = r"```java\s+([\s\S]*?)\s+```"
    matches = [match.strip() for match in re.findall(pattern, text)]
    return matches[0] if len(matches) > 0 else ""

if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="output.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    MODEL_NAME =  args.model_path.split("/")[-1]

    if args.n_gpus > 1: 
        import ray
        ray.init(_temp_dir="/my_local_tmp_dir", log_to_driver=False)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    sampling_params = SamplingParams(
        n=args.n_out_sequences, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_tokens_cot if args.mode == "cot" else args.max_tokens_tir, 
        seed=None if (args.n_out_sequences > 1 and args.mode == "cot") or (args.n_sampling > 1 and args.mode == "tir") else 0
    )

    
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in args.model_path.lower() else "auto",
        quantization="awq_marlin" if "awq" in args.model_path.lower() else None,
        enforce_eager=False,
        max_model_len=args.max_model_len if args.max_model_len > 0 else None,
        trust_remote_code=True,
        tensor_parallel_size=args.n_gpus,
    )

    dataset = load_dataset(args.dataset_path, split="test")

    if args.year_sessions: 
        years_to_consider = args.year_sessions.split(",")
        years_to_consider = [int(el) for el in years_to_consider]
        dataset = dataset.filter(lambda example: example['id'] in years_to_consider)
    
    if args.start_idx > 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, len(dataset)))
    
    if args.max_samples > 0: # to use for debug
        dataset = dataset.select(range(args.max_samples))
    
    SYS_INSTRUCTION = """You are an expert Java developer. Your task is to solve the given university-level Java exam in a single attempt, ensuring that your solution correctly passes all JUnit tests defined in the provided `Test.java` file.  

You may utilize the provided utility Java files as needed. Your implementation should adhere to best coding practices, maintain efficiency, and strictly follow the expected input-output format required to pass the tests."""  


    prompts = []
    for i, item in enumerate(dataset):

        PROMPT_TEMPLATE = """### Utility Files:\n\n{utility_files}\n\n### Test File:\n```java\n\n{test_file}```"""
        
        utilities = item['utility_classes']

        test_content = item['test']['content']

        # correct specific human errors of mispelling on specific test data
        if item['year'] == "2020" and item['session'] == "a05":
            test_content = test_content.replace("createRechargableBattery", "createRechargeableBattery")
            test_content = test_content.replace("createSecureAndRechargableBattery", "createSecureAndRechargeableBattery")

        # fix consistency in package path
        test_content = test_content.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;')
        
        package_line, test_content = extract_and_remove_package_line(java_code=test_content)
        test_file = f"// {item['test']['filename']}\n\n{test_content}"

        utility_files = ""
        for util_class in utilities:
            # fix consistency in package path
            _, util_class_content = extract_and_remove_package_line(util_class['content'].strip())#.replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;').strip()
            utility_files += f"```java\n// {util_class['filename']}\n\n{util_class_content}```\n"

        if "OlympicCoder" in args.model_path:
            
            messages = [
                {"role": "user", "content": SYS_INSTRUCTION.replace('You are an expert Java developer.', '').strip() + "\n\n" + PROMPT_TEMPLATE.format(utility_files=utility_files, test_file=test_file)}
            ]
       
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append({
            "id": f"oop{item['year']}_{item['session']}", 
            "prompt": text.strip(), 
            "package": package_line.strip(),
            "chat_history": messages
        })
        #prompts.append((item['id'], text, messages))
    
    # save first 5 prompts to txt file
    os.makedirs(args.out_dir + "/prompts", exist_ok=True)
    n_prompts_to_stamp = 5 if args.max_samples > 5 else args.max_samples
    with open(args.out_dir + f'/prompts/example_prompts_{MODEL_NAME}.txt', 'w') as f:
        for i in range(n_prompts_to_stamp):
            f.write(f"ID: {prompts[i]['id']}\n")
            f.write(prompts[i]['prompt'])
            f.write("*"*100+'\n')

    logger.info(prompts[0])
    
    batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    logger.info(f"Number of prompts: {len(prompts)}")
    logger.info(f"Number of batches: {len(batches)}")
    logger.info(f"Number of prompts in each batch: {len(batches[0])}")

    
    os.makedirs(args.out_dir + f"/completions/{MODEL_NAME}/{now_dir}", exist_ok=True)
    for id_batch, batch in enumerate(tqdm(batches)):
        ids = [el['id'] for el in batch]
        input_prompts = [el['prompt'] for el in batch]
        packages = [el['package'] for el in batch]

        outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

        for id_out, out in enumerate(outputs):
            completions = [o.text.strip() for o in out.outputs]
            for completion in completions:
                if "</think>" in completion:
                    completion = completion.split("</think>")[1].strip()
                    completion = extract_java_code(completion)
                    completion = packages[id_out] + "\n\n" + completion
                else: 
                    completion = ""
                with open(args.out_dir + f"/completions/{MODEL_NAME}/{now_dir}/completions_{args.mode}.jsonl", 'a') as f:
                    json.dump({"id": ids[id_out], "completion": completion}, f, ensure_ascii=False)
                    f.write('\n')

    """
    if args.n_sampling > 1 and args.mode == "tir":
        import copy
        batches = [[copy.deepcopy(el) for _ in range(args.n_sampling)] for el in prompts]
    else:
        batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    logger.info(f"Number of prompts: {len(prompts)}")
    logger.info(f"Number of batches: {len(batches)}")
    logger.info(f"Number of prompts in each batch: {len(batches[0])}")

    
    os.makedirs(args.out_dir + f"/completions/{MODEL_NAME}", exist_ok=True)
    for id_batch, batch in enumerate(tqdm(batches)):

        if args.mode == "cot":
            ids = [el['id'] for el in batch]
            input_prompts = [el['prompt'] for el in batch]
            gold_answers = [el['answer'] for el in batch]

            outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

            for id_out, out in enumerate(outputs):
                completions = [o.text.strip() for o in out.outputs]
                for completion in completions:
                    with open(args.out_dir + f"/completions/{MODEL_NAME}/completions_{args.mode}.jsonl", 'a') as f:
                        json.dump({"id": ids[id_out], "gold_answer": gold_answers[id_out], "final_answer": extract_answer(completion), "reasoning": completion}, f, ensure_ascii=False)
                        f.write('\n')

        elif args.mode == "tir":
            #print("MODE:", args.mode)
            batch_data = [batch,[],[],[]]
            id_prompt = batch[0]['id']
            gold_answer = batch[0]['answer']
            for n_round in range(args.n_rounds+1):
                input_prompts = [el['prompt'] for el in batch_data[n_round]]
                messages = [el['chat_history'] for el in batch_data[n_round]]
                #print("PROMPTS:", input_prompts)
                outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
                for id_out, out in enumerate(outputs):
                    completion = out.outputs[0].text
                    #print("COMPLETION:", completion)
                    if extract_answer(completion).strip() or n_round == args.n_rounds: # answer found or reached max possible rounds
                        
                        messages[id_out].append({"role": "assistant", "content": completion})
                        if "tora" in args.model_path:
                            text = ""
                            for j, msg in enumerate(messages[id_out]):
                                
                                if msg["role"] == "user":
                                    if j == 0:
                                        msg_content = msg['content'].replace("Solution:", "").strip()
                                        text += f"Question: {msg_content}\n\n"
                                    else:
                                        text += (msg['content'].strip() + "\n")

                                elif msg['role'] == "assistant": 
                                    if j == 1:
                                        text += f"Solution:\n{msg['content'].strip()}\n"
                                    else:
                                        text += (msg['content'].strip() + "\n")
                                
                            logger.info(f"Conversation:\n{text}")
                            logger.info(".....................................\n")
                        
                        with open(args.out_dir + f"/completions/{model_path}/completions_{args.mode}.jsonl", 'a') as f:
                            json.dump({"id": id_prompt, "gold_answer": gold_answer, "final_answer": extract_answer(completion), "messages": messages[id_out]}, f, ensure_ascii=False)
                            f.write('\n')

                    elif "```python" in completion or "deepseek-math" in args.model_path:
                        
                        response = completion.split("```python")[1].split("```")[0] if "```python" in completion else completion.strip()
                        if response.strip():
                            output = exec_code_with_timeout(response, timeout=5)
                            output = tuple(output.values()) if isinstance(output, dict) else output
                            
                        
                        messages[id_out].append({"role": "assistant", "content": completion.strip()})
                        messages[id_out].append({"role": "user", "content": f"```output\n{output.strip()}\n```"})
                        
                        if n_round < args.n_rounds and messages[id_out]:
                            
                            if "tora" in args.model_path:
                                text = ""
                                for j, msg in enumerate(messages[id_out]):
                                    
                                    if msg["role"] == "user":
                                        if j == 0:
                                            msg_content = msg['content'].replace("Solution:", "").strip()
                                            text += f"Question: {msg_content}\n\n"
                                        else:
                                            text += (msg['content'] + "\n")

                                    elif msg['role'] == "assistant": 
                                        if j == 1:
                                            text += f"Solution:\n{msg['content'].strip()}\n"
                                        else:
                                            text += (msg['content'].strip() + "\n")
                                    
                            else:
                                text = tokenizer.apply_chat_template(
                                    messages[id_out],
                                    tokenize=False,
                                    add_generation_prompt=True
                                )
                            
                            batch_data[n_round+1].append({
                                "id": id_prompt,
                                "prompt": text,
                                "chat_history": messages[id_out]}
                            )
    """