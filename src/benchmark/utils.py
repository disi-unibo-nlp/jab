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


def extract_filename(java_code):
    # Matches class, interface, enum, annotation, and record declarations
    pattern = r"(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?(?:class|interface|enum|@interface|record)\s+([A-Za-z_][A-Za-z0-9_]*)"

    match = re.search(pattern, java_code)
    return match.group(1) if match else None

def extract_and_remove_package_line(java_code):
    match = re.search(r'^package\s+[^\n]+;', java_code, flags=re.MULTILINE)
    package_line = match.group(0) if match else None
    cleaned_code = re.sub(r'^package\s+[^\n]+;\n?', '', java_code, flags=re.MULTILINE)
    return package_line, cleaned_code