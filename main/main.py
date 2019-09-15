#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from multiprocessing import Process, Queue
import threading
import os, time, random

sys.path.append('../')
from record import record_start
from operation import operation
from train import sampletrain
from globalvar import *		#导入全局变量


def main():
	# 程序参数处理
	parser = argparse.ArgumentParser(description='knock at')
	parser.add_argument('--mode', help=f'mode: {MODE_ENUM}, default: normal')
	args = parser.parse_args()
	if args.mode:
		mode = args.mode.lower()		#转换为小写
		if mode not in MODE_ENUM:
			print('error: wrong mode!')
			parser.print_help()
		if mode == "train":
			sampletrain.audio_train()
			return

	# 父进程创建Queue，并传给各个子进程：
	q = Queue()
	pRecord = Process(target=record_start, args=(q, mode))
	pCallback = Process(target=operation, args=(q,))
	# 启动子进程pw，写入:
	pRecord.start()
	# 启动子进程pr，读取:
	pCallback.start()

	# while True:
	# 	try:
	# 		pass
	# 	except KeyboardInterrupt:
	# 		exitFlag = 1
	# 		print('等待进程停止...')
		
	# 等待pw结束:
	pRecord.join()
	pCallback.join()

if __name__ == '__main__':
	main()
