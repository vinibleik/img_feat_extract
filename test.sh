#!/bin/bash

if type "python3" >/dev/null 2>&1; then
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	python main.py images/input descriptors --save_img --img_out=images/output
elif type "python" >/dev/null 2>&1; then
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	python main.py images/input descriptors --save_img --img_out=images/output
else
	echo "Python is not installed"
fi
