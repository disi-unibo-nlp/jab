# Java Academic Benchmark (JAB)

This repository contains the code and data for our paper **"Can Large Language Models Pass University Exams on Java Object-Oriented Programming?"**

We introduce the **Java Academic Benchmark (JAB)**, the first **interface-level OOP benchmark for Java**, grounded in authoritative university exams that mirror real-world coding practices.

---

## 📌 Before You Start

If you are interested in reproducing our preprocessing pipelines, follow the guidelines in this section.

> **Note:** The steps below are optional since you can directly download the dataset from our Hugging Face repository. If you only want to use the dataset, skip to the next section. The following steps are intended for reproducibility purposes.

---

## 🔄 Preprocessing

### 1️⃣ Download Exam Data
First, download all the university exams from our [Google Drive repository](https://drive.google.com/drive/folders/1_yFth-GrB8qManGn0GZT5FNidCrhZKBA?usp=sharing).

### 2️⃣ Parse Java Files to JSONL
The raw Java files need to be processed and converted into a JSONL format. To do this, run the [`parse_data.py`](src/preprocessing/parse_data.py) script as follows:

```bash
#!/bin/bash

# Set parameters
EXAMS_DIR="exams"
OUTPUT_PATH="./local_output/jab.jsonl"
EXAM_YEARS=""  # Leave empty to process all years

# Run the Python script
python3 -m src.preprocessing.parse_data \
    --exams_dir "$EXAMS_DIR" \
    --output_path "$OUTPUT_PATH"
```

### 3️⃣ Translate Instructions to English
The exam instructions in each *Test.java* file are written in Italian and need to be translated into English. We use `Gemini-Flash-2.0-001` via the Gemini API for translation.

To reproduce the translation step, run [`translate_test_gemini.py`](src/preprocessing/translate_test_gemini.py):

```bash
#!/bin/bash

# Set default values
MODEL_NAME="gemini-2.0-flash-001"
INPUT_DATA_PATH="local_output/jab.jsonl"
OUTPUT_DATA_PATH="local_output/jab_en.jsonl"

# Run the Python script
python3 -m src.preprocessing.translate_test_gemini \
    --model_name "$MODEL_NAME" \
    --input_data_path "$INPUT_DATA_PATH" \
    --output_data_path "$OUTPUT_DATA_PATH"
```

### 4️⃣ Push Data to Hugging Face
Finally, convert the JSONL file into a Hugging Face Dataset and push it to the Hub using [`push_data_hf.py`](src/preprocessing/push_data_hf.py):

```bash
#!/bin/bash

# Set default values
INPUT_DATA_PATH="local_output/jab_en.jsonl"
OUTPUT_HUB_PATH="<<YOUR_HF_PATH>>"

# Run the Python script
python3 -m src.preprocessing.push_data_hf \
    --input_data_path "$INPUT_DATA_PATH" \
    --output_hub_path "$OUTPUT_HUB_PATH"
```

---

## 🚀 Quick Start

### 🔧 Requirements
Ensure that **Java JDK 21** and **JUnit 5** are installed:

```bash
apt-get update && apt-get install -y openjdk-21-jdk
```

Our experiments were run using the following JUnit version:

```
junit-platform-console-standalone-1.12.1.jar
```

### 📥 Load the Dataset
You can load the dataset from our Hugging Face repository as follows:

```python
from datasets import load_dataset
jab = load_dataset('disi-unibo-nlp/JAB', split="test")
```

---

## 🔍 Sanity Check
Before running experiments, ensure that:
- All Java files are copied locally and can be compiled and executed correctly.
- Your Java environment is set up properly.

Run `sanity_check.py` to verify:

```bash
#!/bin/bash

# Set paths
JUNIT_JAR="lib/junit-platform-console-standalone-1.12.1.jar"
EXAMS_DIR="exams"
DATASET_PATH="disi-unibo-nlp/JAB"

# Run the Python script
python3 sanity_check.py \
    --junit_jar "$JUNIT_JAR" \
    --exams_dir "$EXAMS_DIR" \
    --dataset_path "$DATASET_PATH"
```

### ✅ Expected Output
Check the `sanity_check.log` file. If everything is set up correctly, you should see an output similar to:

```log
INFO - Safe sessions: 103
INFO - All compilations succeeded!
```

If you get this output, you are ready to run all experiments! 🎉

---

## 📬 Contact
For any questions or issues, feel free to open an issue in this repository or reach out to us at
a.cocchieri@unibo.it 😊

