"""
fibomod.py

Description: Basic fibonacci sequence calculation
		base case: 		
		f(1) = 1	
		f(2) = 1
		recursion: 
		f(n) = f(n-1) + f(n-2) , n > 2 

Usage: 	     import fibomod as ...

"""


#import sys
#num_args = len(sys.argv)
#args     = sys.argv

def calc_fibo(n):
	if(n==1 or n==2):
		return 1
	else:
		return calc_fibo(n-1) + calc_fibo(n-2)

#if(num_args > 1):
#	if(int(args[1])>0):
#		answer = calc_fibo(int(args[1]))
#		outStr = "The " + str(args[1]) + "-th Fibonacci number is: " + str(answer)
#		print(outStr) 
#	else:
#		print("Usage: fibo <n>, n > 0")
#
#
#else:	
#	print("Usage: fibo <n>, n > 0")
	

