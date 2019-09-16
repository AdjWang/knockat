#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频文件及数据的输入输出
"""

#导入库
import os
import struct
import wave
import pyaudio
import numpy as np
# from multiprocessing import Pool						#进程并行
from multiprocessing.dummy import Pool as ThreadPool	#线程并发
#导入其他文件

#类
#常量
#全局变量
#函数
def saveaudio(filename, audiodata, channels, format, rate, overwrite=True):
	"""保存声音到文件

		Args:
			filename: 文件目录+文件名
			audiodata: 二进制声音数据 <class 'bytes'>
			channels: 声道数量
			format: 数据格式 例如pyaudio.paInt16
			rate: 采样频率
			overwrite: 如果文件已经存在就: True:覆盖 False:报错

		Returns:
			None

		Raises:
			如果目录不存在，wf = wave.open(filename, 'wb')会报错

		Notes:
			filename示例:
			>>> filename = os.path.join(path, 's5_'+str(savecount)+'.wav')
	"""
	# 保存声音
	# 如果文件名重复，自动重命名
	if os.path.exists(filename) and not overwrite:	#如果文件已经存在
		i = 2
		firstname = '.'.join(filename.split('.')[:-1])			# 提取文件名
		newfirstname = firstname
		fileformat = '.' + filename.split('.')[-1]				# 提取后缀
		while(os.path.isfile(newfirstname + fileformat)):		# 重名文件加括号
			newfirstname = firstname + '({})'.format(i)
			i += 1
		filename = newfirstname + fileformat
	#文件操作
	p = pyaudio.PyAudio()
	wf = wave.open(filename, 'wb')
	wf.setnchannels(channels)
	wf.setsampwidth(p.get_sample_size(format))
	wf.setframerate(rate)
	wf.writeframes(audiodata)
	wf.close()
	print(filename + ' saved')

def newaudio(f, sr, T):
	"""创建音频

		Args:
			f: 音频频率(Hz)
			sr: 采样频率(Hz)
			T: 时间s
		
		Rerutns:
			音频数据，numpy数组格式
	"""
	t = np.linspace(0, T, int(T*sr), endpoint=False)
	x = 0.5*np.sin(2*np.pi*f*t)
	return x

# CHUNK = 1024
def loadwav(filename):
	"""从.wav文件中读取语音

		Args:
			filename: 文件目录+文件名

		Returns:
			归一化数据，numpy数组格式
	"""
	wf = wave.open(filename, 'rb')
	width = wf.getsampwidth()	#数据长度，用于二进制byte数据转换为音频数据时参考
	# read data
	# data = wf.readframes(CHUNK)
	# audiodata = b''
	# while len(data) > 0:
	# 	audiodata += data
	# 	data = wf.readframes(CHUNK)
	audiodata = wf.readframes(wf.getnframes())	#音频二进制数据
	sr = wf.getframerate()		#采样率
	# print(len(audiodata))
	tupdata = struct.unpack('<'+len(audiodata)//width*'h', audiodata)#音频数据byte转int
	floatdata = np.array(tupdata, dtype='float32')
	wf.close()
	del wf
	return floatdata/float(1 << ((8 * width) - 1))		#返回归一化数据

def loaddatafile(path, filetype):
	"""遍历目录下所有指定类型文件

		Args:
			path:数据文件目录
			filetype:文件类型，其余类型不加载

		Returns:
			文件路径列表
	"""
	return [
		os.path.join(path, filename)
		for filename
		in os.listdir(path)
		if filename.endswith(filetype)
		]

def loaddata(datafile):
	"""批量加载目录下音频数据

		Args:
			datafile:数据文件，类型可以是单个文件路径，也可以是多个文件路径列表

		Returns:
			datafile为单个文件：numpy数据
			datafile为目录：数据列表，元素为numpy数据
	"""
	if isinstance(datafile, list):
		pool = ThreadPool(6)	#多线程
		# data = pool.map(lambda x:librosa.load(x, sr=None), datafile)#加载所有数据
		data = pool.map(loadwav, datafile)#加载所有数据
		pool.close()
		pool.join()
	else:
		# data = librosa.load(datafile)
		data = loadwav(datafile)

	return data

#测试程序
if __name__ == '__main__':
	pass