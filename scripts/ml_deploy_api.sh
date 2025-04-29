#!/bin/bash

PROJECT_NAME=$1
if [ -z "$PROJECT_NAME" ]; then
    echo "❌ Usage: $0 <project_name>"
    exit 1
fi

mkdir -p ../$PROJECT_NAME/{data,src,notebooks,deployment,models,tests}

touch ../$PROJECT_NAME/{main.py,README.md,requirements.txt}
touch ../$PROJECT_NAME/src/{train.py,inference.py}
touch ../$PROJECT_NAME/deployment/{app.py,Dockerfile,requirements.txt}
touch ../$PROJECT_NAME/tests/test_inference.py

echo "✅ ML deployment project '$PROJECT_NAME' created."
