#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数字信号处理
"""

#导入库
import sys
import argparse
from multiprocessing import Process, Queue
import threading
import os, time, random

#导入其他文件
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#全局变量
from globalvar import *		#导入全局变量


#类
#常量
#函数
def main():
	"""主函数
	
	启动2个进程 - pRecord, pCallback
	pRecord: 录音(record.py)，数据处理(dataprocess.py)。
	pCallback: 根据处理后的结果执行自定义操作(operation.py)。
	
	pRecord得到敲击位置信息后才执行pCallback，得到敲击位置信息之前pCallback处于阻塞状态。
	敲击位置信息由 q(Queue) 传递。

	Args:
		mode: 主函数参数，用于指定程序运行模式。
			参考globalvar.MODE_ENUM
			MODE_ENUM = ("normal", "sample1", "sample2", "sample3", "sample4", "sample5", "train")
			| normal  | samplen  | train   |
			| 正常处理 | 第n点采样 | 模型训练 |
		
	Returns:
		None
	"""
	# 程序参数处理
	parser = argparse.ArgumentParser(description='knock at')
	parser.add_argument('--mode', help=f'mode: {MODE_ENUM}, default: normal', default='normal')
	args = parser.parse_args()
	if args.mode:
		mode = args.mode.lower()
		if mode not in MODE_ENUM:
			print('error: wrong mode!')
			parser.print_help()
			return
		if mode == "normal":
			from record import record_start
			from operation import operation
			from train import sampletrain
		elif mode == "train":
			from train import sampletrain
			sampletrain.audio_train()
			return
		else:	#sample*
			from record import record_start
			from operation import operation

	# 父进程创建Queue，并传给各个子进程：
	q = Queue()		# 传参
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
