#!/bin/bash

set -e

python3 -m venv venv

source venv/bin/activate

python3 -m pip install --upgrade pip

pip install -U -r requirements.txt

echo -e "\033[1;32m[Environment set up successfully. Use \"source venv/bin/activate\" to enter]\033[0m"