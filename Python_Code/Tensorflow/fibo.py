"""
fibo.py

Description: This computes the fibonacci sequence for multiple inputs
	     using a serial process (one core only)
	     
	     Basic fibonacci sequence calculation
		base case: 		
		f(1) = 1	
		f(2) = 1
		recursion: 
		f(n) = f(n-1) + f(n-2) , n > 2 

Usage: 	     python3 fibo.py <n1> <n2> ... <nK>

"""


import sys
import time
import fibomod as fb

num_args = len(sys.argv)
args     = sys.argv

def check_valid(inlist):
	is_valid = 1
	for i in range(1,len(inlist)):
		if(int(inlist[i])<1):
			is_valid = 0
	return is_valid


if(num_args > 1):
	# Make sure all arguments are valid	
	is_valid = check_valid(args)
	# For all the inputs, sequentially compute the requested Fibonacci number
	if(is_valid):
		t0 = time.time()
		for i in range(1,num_args):
			answer = fb.calc_fibo(int(args[i]))
			outStr = "The " + str(args[i]) + "-th Fibonacci number is: " + str(answer)
			print(outStr)
		
		t1 = time.time()
		wt = t1-t0                
		timeStr = "Wall time = " + str("{:.4}".format(wt))  + " seconds"
		print(timeStr)
	else:
		print("Usage: fibo <n1> <n2> ... <nK>, all inputs integers > 0")	
else:
	print("Usage: fibo <n1> <n2> ... <nK>, all inputs integers > 0")

	

