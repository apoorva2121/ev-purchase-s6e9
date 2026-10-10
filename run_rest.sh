#!/bin/sh
cd "C:/dev/important repo/ml-bench/ev-purchase-s6e9"
mine() { ps -W 2>/dev/null | grep -c "anaconda3/python"; }
echo "waiting; currently $(mine) of ours"
while [ "$(mine)" -gt 1 ]; do sleep 30; done
for s in second_split batch5 seedbag batch6 batch3b batch4 woe mlp_embed mlp; do
  echo "########## $s START $(date +%H:%M:%S) ##########"
  python src/$s.py 2>&1 | grep -v -i "warn"
  echo "########## $s DONE  $(date +%H:%M:%S) ##########"
done
echo "########## ALL DONE ##########"
