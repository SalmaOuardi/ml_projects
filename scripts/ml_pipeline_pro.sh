#!/bin/bash

PROJECT_NAME=$1
if [ -z "$PROJECT_NAME" ]; then
    echo "❌ Usage: $0 <project_name>"
    exit 1
fi

mkdir -p ../$PROJECT_NAME/{data/raw,data/processed,src/features,src/models,src/trainers,src/validators,notebooks,experiments,logs,results}

touch ../$PROJECT_NAME/{main.py,README.md,requirements.txt}
touch ../$PROJECT_NAME/notebooks/data_exploration.ipynb
touch ../$PROJECT_NAME/src/trainers/train_model.py
touch ../$PROJECT_NAME/src/validators/validate_model.py

echo "✅ Advanced ML pipeline project '$PROJECT_NAME' created."
