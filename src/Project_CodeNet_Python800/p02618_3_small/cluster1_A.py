# -*- coding: utf-8 -*-
import sys
import math
from bisect import bisect_left
from bisect import bisect_right
from collections import defaultdict
from heapq import heappop, heappush
import itertools
import random
from decimal import *

input = sys.stdin.readline

def inputInt(): return int(eval(input()))
def inputMap(): return list(map(int, input().split()))
def inputList(): return list(map(int, input().split()))
def inputStr(): return input()[:-1]

inf = float('inf')
mod = 1000000007

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

def main():
	D = inputInt()
	C = inputList()
	S = []
	for i in range(D):
		s = inputList()
		S.append(s)

	ans1 = []
	ans2 = []
	ans3 = []
	ans4 = []
	ans5 = []
	for i in range(D):
		bestSco1 = 0
		bestSco2 = 0
		bestSco3 = 0
		bestSco4 = 0
		bestSco5 = 0
		bestI1 = 1
		bestI2 = 1
		bestI3 = 1
		bestI4 = 1
		bestI5 = 1

		for j,val in enumerate(S[i]):
			if j == 0:
				tmpAns = ans1 + [j+1]
				tmpSco = findScore(tmpAns, S, C)
				if bestSco1 < tmpSco:
					bestSco5 = bestSco4
					bestI5 = bestI4
					bestSco4 = bestSco3
					bestI4 = bestI3
					bestSco3 = bestSco2
					bestI3 = bestI2
					bestSco2 = bestSco1
					bestI2 = bestI1
					bestSco1 = tmpSco
					bestI1 = j+1
			else:
				tmpAns1 = ans1 + [j+1]
				tmpAns2 = ans2 + [j+1]
				tmpAns3 = ans3 + [j+1]
				tmpAns4 = ans4 + [j+1]
				tmpAns5 = ans5 + [j+1]
				tmpSco1 = findScore(tmpAns1, S, C)
				tmpSco2 = findScore(tmpAns2, S, C)
				tmpSco3 = findScore(tmpAns3, S, C)
				tmpSco4 = findScore(tmpAns4, S, C)
				tmpSco5 = findScore(tmpAns5, S, C)
				if bestSco1 < tmpSco1:
					bestSco5 = bestSco4
					bestI5 = bestI4
					bestSco4 = bestSco3
					bestI4 = bestI3
					bestSco3 = bestSco2
					bestI3 = bestI2
					bestSco2 = bestSco1
					bestI2 = bestI1
					bestSco1 = tmpSco1
					bestI1 = j+1
				if bestSco1 < tmpSco2:
					bestSco5 = bestSco4
					bestI5 = bestI4
					bestSco4 = bestSco3
					bestI4 = bestI3
					bestSco3 = bestSco2
					bestI3 = bestI2
					bestSco2 = bestSco1
					bestI2 = bestI1
					bestSco1 = tmpSco1
					bestI1 = j+1
				if bestSco1 < tmpSco3:
					bestSco5 = bestSco4
					bestI5 = bestI4
					bestSco4 = bestSco3
					bestI4 = bestI3
					bestSco3 = bestSco2
					bestI3 = bestI2
					bestSco2 = bestSco1
					bestI2 = bestI1
					bestSco1 = tmpSco1
					bestI1 = j+1
				if bestSco1 < tmpSco4:
					bestSco5 = bestSco4
					bestI5 = bestI4
					bestSco4 = bestSco3
					bestI4 = bestI3
					bestSco3 = bestSco2
					bestI3 = bestI2
					bestSco2 = bestSco1
					bestI2 = bestI1
					bestSco1 = tmpSco1
					bestI1 = j+1
				if bestSco1 < tmpSco5:
					bestSco5 = bestSco4
					bestI5 = bestI4
					bestSco4 = bestSco3
					bestI4 = bestI3
					bestSco3 = bestSco2
					bestI3 = bestI2
					bestSco2 = bestSco1
					bestI2 = bestI1
					bestSco1 = tmpSco1
					bestI1 = j+1
		ans1.append(bestI1)
		ans2.append(bestI2)
		ans3.append(bestI3)
		ans4.append(bestI4)
		ans5.append(bestI5)

	for i in ans1:
		print(i)


def findScore(ans, S, C):
	scezhu = [inf for i in range(26)]
	sco = 0
	for i,val in enumerate(ans):
		tmp = S[i][val-1]
		scezhu[val-1] = i
		mins = 0
		for j,vol in enumerate(C):
			if scezhu[j] == inf:
				mins = mins + (vol * (i+1))
			else:
				mins = mins + (vol * ((i+1)-(scezhu[j]+1)))
		tmp -= mins
		sco += tmp
	return sco

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":
	main()
