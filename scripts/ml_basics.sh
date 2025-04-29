#!/bin/bash

PROJECT_NAME=$1
if [ -z "$PROJECT_NAME" ]; then
    echo "❌ Usage: $0 <project_name>"
    exit 1
fi

mkdir -p ../$PROJECT_NAME/{data/raw,data/processed,notebooks,src,models,results}

touch ../$PROJECT_NAME/{main.py,README.md,requirements.txt}
touch ../$PROJECT_NAME/notebooks/01_exploration.ipynb
touch ../$PROJECT_NAME/src/{config.py,data_preprocessing.py,train_model.py,evaluate_model.py}

echo "✅ Basic ML project '$PROJECT_NAME' created."
