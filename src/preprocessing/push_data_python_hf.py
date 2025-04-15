import os
from datasets import Dataset
import pandas as pd

sessions = os.listdir('exams_python/oop2023')
dataset = []
for session in sessions:

    item = {"year": "2023", "session": session, "utility_classes": {"filename": "test.py", "content": ""}, "test": {"filename": "test.py", "content": ""}, "solution": {"filename": "test.py", "content": ""}}
    with open(f'exams_python/oop2023/{session}/test.py') as f:
        content = f.read().strip()

        utility = content.split("# Solution")[0].replace('# Utility Files','').strip()
        solution = content.split("# Solution")[1].split("# Test File")[0].strip()
        test = content.split("# Test File")[1].strip()

        print("-------- UTILITY ------------")
        print(utility)
        item['utility_classes']['content'] = utility

        print("-------- SOLUTION ------------")
        
        item['solution']['content'] = solution
        print(item['solution']['content'])

        print("-------- TEST ------------")
        print(test)
        item['test']['content'] = test
    
    dataset.append(item)

df = pd.DataFrame(dataset)

ds = Dataset.from_pandas(df)
print(ds)

ds.push_to_hub("disi-unibo-nlp/JAB-python", split="test", private=True)
