#!/bin/bash

PROJECT_NAME=$1
if [ -z "$PROJECT_NAME" ]; then
    echo "❌ Usage: $0 <project_name>"
    exit 1
fi

mkdir -p ../$PROJECT_NAME/{data,notebooks,src,outputs/models,outputs/figures,logs}

touch ../$PROJECT_NAME/{main.py,README.md,requirements.txt}
touch ../$PROJECT_NAME/notebooks/01_data_exploration.ipynb

# General-purpose DL source files
touch ../$PROJECT_NAME/src/{model.py,train.py,test.py,dataset.py,utils.py,config.py}

echo "✅ General Deep Learning project '$PROJECT_NAME' created."
