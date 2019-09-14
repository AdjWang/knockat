#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pygame
import sys
import matplotlib.pyplot as plt

from globalvar import *		#导入全局变量

def operation(q):
	'''
	'''
	global exitFlag

	poslist = [0]*5
	while not exitFlag:  #
		value = q.get(True)
		print(value)

		poslist[value] = 1
		x = [1, 2, 3, 4, 5]
		plt.ion()
		plt.clf()
		plt.yticks([0, 1])
		plt.bar(x, poslist)
		plt.pause(0.3)

		poslist[value] = 0
		plt.ion()
		plt.clf()
		plt.yticks([0, 1])
		plt.bar(x, poslist)
		plt.pause(0.01)
