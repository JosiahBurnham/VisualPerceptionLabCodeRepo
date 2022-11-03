#!/bin/bash

lambda=(0.001 0.01 0.1 1 10 100)
holdout=(0 1 2 3 4 5 6 7 8 9)
for i in "${lambda[@]}"; do
	for j in "${holdout[@]}"; do	
		python3 testarg.py $i $j
	done
done
