#!/bin/sh
cd "C:/dev/important repo/ml-bench/ev-purchase-s6e9"
mine() { ps -W 2>/dev/null | grep -c "anaconda3/python"; }
while [ "$(mine)" -gt 1 ]; do sleep 30; done
for s in batch4 woe; do
  echo "########## $s START $(date +%H:%M:%S) ##########"
  python -u src/$s.py 2>&1
  echo "########## $s DONE  $(date +%H:%M:%S) ##########"
done
echo "########## TAIL QUEUE DONE ##########"
