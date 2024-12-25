#!/bin/bash

set -e
echo "create venv"
python -m venv venv || {echo"failed"; exit 1;}

echo "Activate venv"
source venv/bin/activate || {echo "failed to activate venv"; exit 1;}

echo "install deps"
pip install -r requirements.txt || {}

