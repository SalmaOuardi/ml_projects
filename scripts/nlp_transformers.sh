#!/bin/bash

PROJECT_NAME=$1
if [ -z "$PROJECT_NAME" ]; then
    echo "❌ Usage: $0 <project_name>"
    exit 1
fi

mkdir -p ../$PROJECT_NAME/{data,src/models,src/preprocessing,src/tokenizer,notebooks,logs,results}

touch ../$PROJECT_NAME/{main.py,README.md,requirements.txt}
touch ../$PROJECT_NAME/src/{train.py,evaluate.py}
touch ../$PROJECT_NAME/src/models/transformer_model.py
touch ../$PROJECT_NAME/src/preprocessing/clean_text.py
touch ../$PROJECT_NAME/src/tokenizer/tokenize.py
touch ../$PROJECT_NAME/notebooks/text_analysis.ipynb

echo "✅ NLP Transformers project '$PROJECT_NAME' created."
