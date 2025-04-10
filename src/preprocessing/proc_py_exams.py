import json
import os
with open('completions/python_conversion/2025-04-09_15-34-24/converted_exams_python.jsonl') as f:
    data = [json.loads(line) for line in f.readlines()]

os.makedirs('exams_python', exist_ok=True)

for item in data:
    year = item['year']
    session = item['session']
    out_dir = f'exams_python/oop{year}_v2/{session}'
    os.makedirs(out_dir, exist_ok=True)
    
    with open(out_dir + "/test.py", 'w') as f:
        f.write(item['python_code'])