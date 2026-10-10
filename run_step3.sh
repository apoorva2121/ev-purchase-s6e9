#!/bin/sh
cd "C:/dev/important repo/ml-bench/ev-purchase-s6e9"
while [ "$(ps -W 2>/dev/null | grep -c 'anaconda3/python')" -gt 2 ]; do sleep 30; done
python -u src/step3_te_all.py
