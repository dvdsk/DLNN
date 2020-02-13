#!/usr/bin/bash

if ! mypy src/main.py; then
    read -p "Type errors detected, Press any key to continue" -n 1 -s
fi

if ! black src/*; then
    read -p "Could not auto format, Press any keye to continue" -n 1 -s
fi

if ! flake8 src/*.py --ignore=E501,E302,E303,E231; then
    read -p "Lint errors detected, Press any key to continue" -n 1 -s
fi

if ! pytest src/*.py; then
    echo "Errors found during testing" 
fi
