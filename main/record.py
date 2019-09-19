#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).
"""

# import copy
import pyaudio
import numpy as np

from dataprocess import dataprocess
from globalvar import *		#导入全局变量


# RECORD_SECONDS = 5
def record_start(callback = None, mode = "normal"):
	"""加载麦克风录音

	Args:
		callback: 数据处理完成后的操作，函数或Queue。Queue用于给operation.py进程传参
		mode: 执行模式，详见globalvar.py, MODE_ENUM
			| normal  | samplen  | train   |
			| 正常处理 | 第n点采样 | 模型训练 |

	Returns:
		None
	"""
	global exitFlag
	
	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,#format=p.get_format_from_width(WIDTH),
					channels=CHANNELS,
					rate=RATE,
					input=True,
					# output=True,
					frames_per_buffer=CHUNK)

	print("开始录音...(按下Ctrl+C中断)")

	# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	# 	data = stream.read(CHUNK)
	# 	stream.write(data, CHUNK)		#播放音频流

	try:
		while not exitFlag:
			# for i in range(0, CHUNK_NUM):
			# 	data = stream.read(CHUNK)
			# 	# stream.write(data, CHUNK)		#播放音频流
			# 	frames1.append(data)
			audio_data = stream.read(CHUNK*CHUNK_NUM)
			# audio_data = np.fromstring(data, dtype=np.short)
			""" 数据处理程序 """
			# print('数据处理...')
			dataprocess(audio_data, callback, mode)
	except KeyboardInterrupt:
		exitFlag = 1

	print("停止录音")
	# print(len(audio_data))
	# stream.write(audio_data, len(audio_data)//WIDTH)		#播放音频流

	stream.stop_stream()
	stream.close()

	p.terminate()


""" 测试代码 """
if __name__ == '__main__':
	def trigger(position):
		"""	敲击后触发该函数
		"""
		print(position)

	record_start(callback = trigger)
