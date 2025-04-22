from openai import OpenAI

import os
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from utils import extract_java_code, extract_filename, extract_and_remove_package_line
from datasets import load_dataset
# parse input args
   

@dataclass
class ScriptArguments:
    batch_id: Optional[str] = field(default="batch_6806ba80bad88190b05b2ad1b62de07a", metadata={"help": "batch id to retrieve from OpenAI API"})
    out_dir: Optional[str] = field(default="out/completions/gpt-4.1-2025-04-14/cot/pass1/2025-04-21_21-36-59", metadata={"help": "directory where to store results."})

if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    dataset = load_dataset("disi-unibo-nlp/JAB", split="test")
    mode = args.out_dir.split("/")[3]
    #print(client.batches.retrieve(args.batch_id))
    response = client.batches.retrieve(args.batch_id)
    print(response)
    if response.status =='completed':
        print("INFERENCE COMPLETED!")
        out_file_id = response.output_file_id
        
        file_response = client.files.content(out_file_id)
        print("Saving results...")
        for line in file_response.text.splitlines():
            with open(f'{args.out_dir}/raw_completions.jsonl', 'a') as f:
                json.dump(json.loads(line), f, ensure_ascii=False)
                f.write('\n')

            result = json.loads(line)
            completion = result['response']['body']['choices'][0]['message']['content']
            
            
            id_exam = result['custom_id'].split('-')[1].strip()
            year = id_exam.split('_')[0].replace("oop", "").strip()
            session = id_exam.split('_')[1]
            item_dataset = dataset.filter(lambda x: x['year'] == str(year) and x['session'] == str(session))
            
            test_content = item_dataset[0]['test']['content']
            package_line, _ = extract_and_remove_package_line(test_content)
            java_codes = extract_java_code(completion)

            java_codes = [package_line.strip() + "\n\n" + code for code in java_codes]
    
            java_codes = [{'filename': extract_filename(java_code), "content": java_code} for java_code in java_codes]

            with open(f"{args.out_dir}/completions_{mode}.jsonl", 'a') as f:
                res_dict = {"id": id_exam, "code": java_codes, "completion": completion}

                json.dump(res_dict, f, ensure_ascii=False)
                f.write('\n')

        print("Done!")
    else:
        print("BATCH STILL PROCESSING...")
        print(f"STATUS: {response.status}")