import argparse
from collections import defaultdict
from datasets import load_dataset
from src.benchmark.utils import extract_and_remove_package_line
import lizard
import re

SYS_INSTRUCTION = """You are an expert Java developer. Your task is to solve the given university-level Java exam in a single attempt, ensuring that your solution correctly passes all JUnit tests defined in the provided `Test.java` file.

You may use the provided utility Java files as needed. Your final answer must consist of one or more Java scripts, each enclosed separately between ```java and ``` in different code blocks.
"""

def build_prompt(item, args, tokenizer):
    package_line = ""
    if "python" not in args.dataset_path:
        PROMPT_TEMPLATE = """### Utility Files:\n\n{utility_files}\n\n### Test File:\n```java\n\n{test_file}```"""

        utilities = item['utility_classes']
        test_content = item['test']['content']

        package_line, test_content = extract_and_remove_package_line(java_code=test_content)
        test_file = f"// {item['test']['filename']}\n\n{test_content}"

        utility_files = ""
        for util_class in utilities:
            _, util_class_content = extract_and_remove_package_line(util_class['content'].strip())
            utility_files += f"```java\n// {util_class['filename']}\n\n{util_class_content}```\n"

        prompt = PROMPT_TEMPLATE.format(utility_files=utility_files, test_file=test_file)

    else:
        PROMPT_TEMPLATE = """```python\n\n### Utility\n\n{utility}\n\n### Test\n\n{test}```"""
        utilities = item['utility_classes']
        test_content = item['test']['content']
        prompt = PROMPT_TEMPLATE.format(utility=utilities, test=test_content)

    if "Qwen2.5-Coder" in args.model_path:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": SYS_INSTRUCTION.replace('You are an expert Java developer.', '').strip() + "\n\n" + prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": SYS_INSTRUCTION},
            {"role": "user", "content": prompt}
        ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

def calculate_cognitive_complexity(code, language="java"):
    """
    Calculate cognitive complexity based on SonarQube's methodology
    
    Simplified implementation focusing on:
    1. Increments for control flow structures (if, for, while, switch, etc.)
    2. Additional increments for nesting
    3. Increments for breaks in logical flow (break, continue, goto)
    4. Increments for logical operators that add complexity
    """
    # Clean up the code (remove comments, string literals)
    # This is a simplified version; a real implementation would handle these cases better
    clean_code = re.sub(r'//.*?$|/\*.*?\*/|".*?"|\'.*?\'', '', code, flags=re.MULTILINE|re.DOTALL)
    
    complexity = 0
    nesting_level = 0
    
    # Patterns for control flow structures
    control_flow_patterns = [
        r'\bif\s*\(', r'\belse\s+if\s*\(', r'\belse\b',
        r'\bfor\s*\(', r'\bwhile\s*\(', r'\bdo\b',
        r'\bswitch\s*\(', r'\bcase\b', r'\bdefault\b',
        r'\bcatch\s*\('
    ]
    
    # Patterns for logical operators that increase complexity
    logical_operators = [r'&&', r'\|\|']
    
    # Patterns for breaks in logical flow
    flow_breaks = [r'\bbreak\b', r'\bcontinue\b', r'\breturn\b']
    
    # Process the code line by line for better tracking
    lines = clean_code.split('\n')
    
    for line in lines:
        # Check for opening braces - indicating nesting level increase
        open_braces = line.count('{')
        close_braces = line.count('}')
        
        # Update nesting level
        nesting_level += open_braces - close_braces
        
        # Check for control flow structures
        for pattern in control_flow_patterns:
            matches = re.findall(pattern, line)
            for _ in matches:
                # Base increment for control structure
                complexity += 1
                # Additional increment for nesting (except first level)
                if nesting_level > 0:  # In SonarQube, we'd add nesting_level increments
                    complexity += min(nesting_level, 3)  # Cap at level 3 for simplicity
        
        # Check for logical operators
        for pattern in logical_operators:
            matches = re.findall(pattern, line)
            complexity += len(matches)
        
        # Check for breaks in logical flow
        for pattern in flow_breaks:
            matches = re.findall(pattern, line)
            complexity += len(matches)
    
    return complexity

def count_code_lines_and_complexity(item, key, args):
    code_lines = []
    full_code = ""

    if "python" not in args.dataset_path:
        
        if key == "solution":
            content = "\n\n".join([el['content'] for el in item[key]])
        else:
            content = item[key]['content']

        _, content_clean = extract_and_remove_package_line(java_code=content)
        code_lines.extend(content_clean.strip().splitlines())
        full_code += content_clean + "\n"
        
        if key != "solution":
            for util_class in item['utility_classes']:
                _, util_content_clean = extract_and_remove_package_line(util_class['content'].strip())
                code_lines.extend(util_content_clean.strip().splitlines())
                full_code += util_content_clean + "\n"

    # else:
    #     sol_content = "\n\n".join([el['content'] for el in item[key]])
    #     code_lines.extend(sol_content.strip().splitlines())
    #     full_code += sol_content + "\n"


    code_lines = [line for line in code_lines if line.strip() != ""]

    total_cyclomatic = -1
    total_cognitive = -1
    if key == "solution":
        # Analyze cyclomatic complexity with lizard
        analysis = lizard.analyze_file.analyze_source_code("dummy.java", full_code)
        total_cyclomatic = sum(func.cyclomatic_complexity for func in analysis.function_list)

        # Calculate cognitive complexity
        language = "python" if "python" in args.dataset_path else "java"
        total_cognitive = calculate_cognitive_complexity(full_code, language)

    return len(code_lines), total_cyclomatic, total_cognitive

def main():
    parser = argparse.ArgumentParser(description="Count average tokens, LOC, and complexity per year in JAB dataset")
    parser.add_argument("--dataset_path", type=str, default="disi-unibo-nlp/JAB", help="Path or name of dataset")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct", help="Path to model")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset(args.dataset_path)
    print(dataset)

    token_counts = defaultdict(int)
    token_counts_out = defaultdict(int)
    loc_counts = defaultdict(int)
    loc_counts_out = defaultdict(int)
    cyclomatic_counts = defaultdict(int)
    cognitive_counts = defaultdict(int)
    prompt_counts = defaultdict(int)

    for i, item in enumerate(dataset['test']):
        prompt_text = build_prompt(item, args, tokenizer)
        tokens = tokenizer(prompt_text, add_special_tokens=False)
        num_tokens = len(tokens['input_ids'])
        solution = [el['content'] for el in item['solution']]
        tokens_out = tokenizer("\n\n".join(solution), add_special_tokens=False)
        num_tokens_out = len(tokens_out['input_ids'])

        num_loc, _, _ = count_code_lines_and_complexity(item, 'test', args)
        num_loc_out, num_cyclomatic, num_cognitive = count_code_lines_and_complexity(item, 'solution', args)


        year = item.get('year', 'unknown')
        token_counts[year] += num_tokens
        token_counts_out[year] += num_tokens_out
        loc_counts[year] += num_loc
        loc_counts_out[year] += num_loc_out
        cyclomatic_counts[year] += num_cyclomatic
        cognitive_counts[year] += num_cognitive
        prompt_counts[year] += 1

        print(f"[{i+1}/{len(dataset['test'])}] Year: {year} → {num_tokens_out} tokens, {num_loc_out} LOC, "
              f"{num_cyclomatic} cyclomatic, {num_cognitive} cognitive")

    print("\n=== INPUT: Averages Per Year ===")
    for year in sorted(prompt_counts.keys()):
        num_prompts = prompt_counts[year]
        avg_tokens = token_counts[year] / num_prompts if num_prompts > 0 else 0
        avg_loc = loc_counts[year] / num_prompts if num_prompts > 0 else 0
        print(f"{year}: {avg_tokens:.2f} tokens, {avg_loc:.2f} LOC, (prompts: {num_prompts})")
              #f"{avg_cyclomatic:.2f} cyclomatic, {avg_cognitive:.2f} cognitive (prompts: {num_prompts})")
    
    print("\n=== OUTPUT: Averages Per Year ===")
    for year in sorted(prompt_counts.keys()):
        num_prompts = prompt_counts[year]
        avg_tokens_out = token_counts_out[year] / num_prompts if num_prompts > 0 else 0
        avg_loc_out = loc_counts_out[year] / num_prompts if num_prompts > 0 else 0
        avg_cyclomatic = cyclomatic_counts[year] / num_prompts if num_prompts > 0 else 0
        avg_cognitive = cognitive_counts[year] / num_prompts if num_prompts > 0 else 0
        print(f"{year}: {avg_tokens_out:.2f} tokens, {avg_loc_out:.2f} LOC, "
        f"{avg_cyclomatic:.2f} cyclomatic, {avg_cognitive:.2f} cognitive (prompts: {num_prompts})")
    

if __name__ == "__main__":
    main()