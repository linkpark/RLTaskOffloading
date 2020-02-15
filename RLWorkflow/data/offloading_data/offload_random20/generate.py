import os
import sys
import random


fat = [0.1, 0.3, 0.5, 0.7, 0.9]
density = [0.5, 0.6, 0.7, 0.9]
regularity = [0.5, 0.7, 0.9]
ccr = [0.3, 0.4, 0.5]

mindata = int(5 * (1 * 1024.0 * 1024.0 ))
maxdata = int(50 * (1 * 1024.0 * 1024.0 ))


def main():
	for i in range(1000):
		command = './daggen --dot -n 20'
		file_name = 'random.20.' + str(i) +'.gv'
		command = (command + ' --ccr ' + str(ccr[random.randint(0,len(ccr)-1)])  +
				' --fat ' +  str(fat[random.randint(0,len(fat)-1)])  +
				' --regular ' + str(regularity[random.randint(0,len(regularity)-1)])  +
				' --density ' + str(density[random.randint(0,len(density)-1)]) +
                ' --mindata ' + str(mindata) +
                ' --maxdata ' + str(maxdata))
		print(command)

		os.system( command + '> '+file_name)

if __name__ == '__main__':
	main()

