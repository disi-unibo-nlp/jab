import os
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import argparse

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Assert that HF_TOKEN is set
assert HF_TOKEN, "HF_TOKEN is required."

login(token=HF_TOKEN)

def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description="Process dataset and push to Hugging Face Hub.")
    
    # Input and output arguments
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to input dataset (JSONL).")
    parser.add_argument("--output_hub_path", type=str, required=True, help="Hugging Face Hub repository to push the dataset to.")
    
    return parser.parse_args()

def process_dataset(input_data_path):
    """
    Processes the dataset: reads from JSONL, converts to DataFrame, cleans columns.
    """
    with open(input_data_path, 'r') as f:
        dataset = [json.loads(line) for line in f.readlines()]

    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    dataset = Dataset.from_pandas(df)

    new_dataset = []
    for item in tqdm(dataset, desc="Processing items"):

        for idx in range(len(item['exam'])):
            item['exam'][idx]['content'] = item['exam'][idx]['content'].strip().replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;')

        # correct specific human errors of mispelling
        if item['year'] == 2020 and item['session'] == "a05":
            item['test']['content'] = item['test']['content'].replace("createRechargableBattery", "createRechargeableBattery")
            item['test']['content'] = item['test']['content'].replace("createSecureAndRechargableBattery", "createSecureAndRechargeableBattery")
            
            for idx in range(len(item['solution'])):
                item['solution'][idx]['content'] = item['solution'][idx]['content'].replace("createRechargableBattery", "createRechargeableBattery")
                item['solution'][idx]['content'] = item['solution'][idx]['content'].replace("createSecureAndRechargableBattery", "createSecureAndRechargeableBattery")

        item['test']['content'] = item['test']['content'].replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;')
        
        for idx in range(len(item['solution'])):
            item['solution'][idx]['content'] = item['solution'][idx]['content'].replace('.e1;','.sol1;').replace('.sol2;','.sol1;').replace('.e2;','.sol1;')

        new_dataset.append(item)
    
    df = pd.DataFrame(new_dataset)
    
    # Convert 'year' column to string
    df["year"] = df["year"].astype(str)
    
    # Convert to Hugging Face Dataset
    ds = Dataset.from_pandas(df)
    
    # Remove unwanted column
    ds = ds.remove_columns("test_ita")
    
    # Rename column 'exam' to 'utility_classes'
    ds = ds.rename_column("exam", "utility_classes")
    
    return ds

def main():
    """
    Main function to process the dataset and push it to Hugging Face Hub.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Process the dataset
    ds = process_dataset(args.input_data_path)
    
    # Push to Hugging Face Hub
    ds.push_to_hub(args.output_hub_path, split="test")
    
    print(f"Dataset pushed to {args.output_hub_path} successfully!")

if __name__ == "__main__":
    main()
