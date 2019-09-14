#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import Process, Queue
import threading
import os, time, random

from record import record_start
from operation import operation
from globalvar import *		#导入全局变量

if __name__=='__main__':
	# 父进程创建Queue，并传给各个子进程：
	q = Queue()
	pRecord = Process(target=record_start, args=(q,))
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


