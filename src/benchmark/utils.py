import os
import pandas as pd
import numpy as np 
import subprocess
import re




def extract_java_code(text):
    """Extracts Java code blocks enclosed between ```java and ```."""
    pattern = r"```java\s+([\s\S]*?)\s+```"
    matches = [match.strip() for match in re.findall(pattern, text)]
    return matches if len(matches) > 0 else ""

def extract_python_code(text):
    """Extracts python code blocks enclosed between ```python and ```."""
    pattern = r"```python\s+([\s\S]*?)\s+```"
    matches = [match.strip() for match in re.findall(pattern, text)]
    return matches[0] if len(matches) > 0 else ""


# def extract_filename(java_code):
#     # Matches class, interface, enum, annotation, and record declarations
#     pattern = r"(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?(?:class|interface|enum|@interface|record)\s+([A-Za-z_][A-Za-z0-9_]*)"

#     match = re.search(pattern, java_code)
#     return match.group(1) if match else None

def extract_filename(java_code):
    # Remove multi-line comments first
    code_without_comments = re.sub(r'/\*[\s\S]*?\*/', '', java_code)
    
    # Match class declarations with capital-letter class names
    pattern = r"(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?(?:class|interface|enum|@interface|record)\s+([A-Z][A-Za-z0-9_]*)"
    
    match = re.search(pattern, code_without_comments)
    return match.group(1) if match else None

def extract_and_remove_package_line(java_code):
    match = re.search(r'^package\s+[^\n]+;', java_code, flags=re.MULTILINE)
    package_line = match.group(0) if match else None
    cleaned_code = re.sub(r'^package\s+[^\n]+;\n?', '', java_code, flags=re.MULTILINE)
    return package_line, cleaned_code

def check_mandatory_tests(item, conditions):
    cond_key = f"oop{item['year']}_{item['session']}"
    
    if item["compilation_passed"] and not item["runtime_passed"]:
        if not "fails" in item["runtime_errors"][0]:
            # exception during previous execution
            item["runtime_passed_mandatory"] = False
            return item 

        fails = item["runtime_errors"][0]['fails']
        logger.info(f"Fail methods: {fails}")
        if not fails:
            item["runtime_passed_mandatory"] = True
            return item

        if not cond_key in conditions:
            if all('optional' in test.lower() for test in fails):
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False

        elif cond_key == "oop2018_a06":
            if ("testOne" not in fails and "testOneOf" not in fails and "testZeroOrMany" not in fails and "testSequence" not in fails) or ("testFullSequence" not in fails):
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False

        elif cond_key == "oop2022_a02a":
            if len(fails) == 1 and "testFromList" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        
        elif cond_key == "oop2022_a03a":
            if len(fails) == 1 and "testFinitePossibilities" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False

        elif cond_key == "oop2022_a02b": 
            if len(fails) == 1 and "testFromNonEmptyList" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        
        elif cond_key == "oop2022_a03b": 
            if len(fails) == 1 and "testConstant" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        
        elif cond_key == "oop2022_a04":
            if len(fails) == 1 and "testBasic" not in fails:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        
        elif cond_key == "oop2021_a04":
            if len(fails) == 1 and fails[0] in "testFold/testFlatMap/testFilter":
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        elif cond_key == "oop2024_a01a":
            if len(fails) == 1 and fails[0] in "testOr/testSeq":
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False
        else:
            if len(fails) == 1:
                item["runtime_passed_mandatory"] = True
            else:
                item["runtime_passed_mandatory"] = False

    elif item["compilation_passed"] and item["runtime_passed"]:
        item["runtime_passed_mandatory"] = True
    else:
        item["runtime_passed_mandatory"] = False
    
    return item