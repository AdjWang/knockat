#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频文件及数据的输入输出
"""

#导入库
import os
import math
import struct
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
# import webrtcvad
import librosa
from librosa import display
from scipy.fftpack import fft
from sklearn.externals import joblib

#导入其他文件
import audioio as io
import dsp
from globalvar import *		#导入常量

#类
#常量
#全局变量
audio_cache = {}	#记录声音数据，用于连续录音
module_path = './modules/'	#模型文件所在文件夹位置
if os.path.exists(os.path.join(module_path, 'pca.m')):
	pca = joblib.load(os.path.join(module_path, 'pca.m'))		#读取模型
if os.path.exists(os.path.join(module_path, 'svm.m')):
	classifier = joblib.load(os.path.join(module_path, 'svm.m'))		#读取模型
# coef = np.load('../modules/coef.npy')
fir = dsp.FIR_Filter(44100, 300, 330, deltp=1.008, delts=0.005)		#FIR滤波器初始化

#函数
# def readmodel(filename):
# 	"""加载模型文件

# 	Args:
# 		filename: 模型文件
	
# 	Rerutns:
# 		模型
# 	"""
# 	return joblib.load(filename)

def diswave(x, sr, ion=False):
	"""绘制波形幅度包络图

	Args:
		x: 波形数据
		sr: 采样频率
		ion: pyplot显示模式，默认阻塞模式

	Returns:
		None
	"""
	if ion:
		plt.ion()
		plt.clf()

	display.waveplot(x, sr)

	if ion:
		plt.pause(0.01)
	if not ion:
		plt.show()

def dataprocess(audiodata, callback = None, mode = "normal"):
	"""数据处理

	流程：
		if有缓存，说明上次检测到一部分声音，本次开头一定是剩余声音，不进行端点检测，直接连接:
			连接
			清空缓存
			数据处理(def process(floatdata):)
		else没有缓存，正常处理:
			端点检测
			if右端点超出范围:
				数据添加到缓存,结束本次数据处理(return)
			elif端点都在当前帧内，正常截取:
				截取数据
				数据处理(def process(floatdata):)
			else:
				pass
		
	Args:
		audiodata: 二进制音频数据
		callback: 数据处理完成后的操作，函数或Queue。Queue用于给operation.py进程传参
		mode: 执行模式，详见globalvar.py, MODE_ENUM
			| normal  | samplen  | train   |
			| 正常处理 | 第n点采样 | 模型训练 |

	Returns:
		None
	"""
	def process(floatdata):
		"""数据处理

		不同位置用到2次，所以单独写成函数

		Args:
			floatdata: PCM音频数据 numpy-float32类型

		Returns:
			None
		"""
		# floatdata, sr = librosa.load('./test.wav')	#读取声音

		diswave(floatdata, RATE, ion=True)		#绘制波形
		
		# filtered_data = fir.filtering(floatdata)							#FIR滤波，得到特征数据
		filtered_data = floatdata

		n_fft = 512
		mfcc = dsp.MFCC_Filter(44100, n_fft)
		mfcc_filter = mfcc.make()
		ym = dsp.rfft(filtered_data, n_result=n_fft)**2 					#能量谱
		featuredata = mfcc.filtering(ym, mfccfilter=mfcc_filter)		#mfcc

		# reduceddata = pca.transform(featuredata.reshape(1, -1))			#PCA降维，模型需要数据格式转换
		reduceddata = featuredata.reshape(1, -1)

		probability = classifier.predict_proba(reduceddata)	#分类器输出概率
		position = np.argmax(probability)					#概率最大的位置
		print(probability)
		print(position)

		# print(position, max(probability))
		if callback:				#用户定义的后续操作
			if callable(callback):		#是一个函数
				callback(position)
			elif isinstance(callback, multiprocessing.queues.Queue):	#进程通信队列
				callback.put(position, block=False)

	def save_audio(audiodata):
		"""保存音频，用于samplen模式

		不同位置用到2次，所以单独写成函数
		保存位置在该函数内定义
		其余音频参数参考globalvar.py

		Args:
			audiodata: 二进制音频数据

		Returns:
			None
		"""
		dirname = os.path.join(os.path.dirname(__file__), f'../samples/{mode}/')
		filename = os.path.join(os.path.dirname(__file__), f'../samples/{mode}/{mode}.wav')				#io.saveaudio函数自动重命名
		if not os.path.exists(dirname):		#如果sample文件夹不存在，创建文件夹
			os.makedirs(dirname)
		io.saveaudio(filename, audiodata, CHANNELS, FORMAT, RATE, overwrite=False)		#保存音频文件，函数自动重命名

	tupdata = struct.unpack('<'+len(audiodata)//WIDTH*'h', audiodata)#音频数据byte转int
	floatdata = np.array(tupdata, dtype='float32')/float(1 << ((8 * WIDTH) - 1))		#归一化数据

	if len(audio_cache) > 0:	#有缓存，需要连接剩下的声音
		if mode[:-1] == "sample":
			save_audio(audio_cache['byte_audio']+audiodata[:math.ceil(WIDTH*audio_cache['endpoint'])])#保存声音
			audio_cache.clear()		#清空缓存
			return

		floatdata = np.concatenate([audio_cache['last_audio'], floatdata[:audio_cache['endpoint']]])#连接上次和本次数据
		audio_cache.clear()		#清空缓存
		#数据处理
		process(floatdata)
	else:						#没有缓存，正常处理
		l, r = dsp.vad(floatdata, GATE, SIGNALLEN)		#端点检测，检测敲击左右端点
		if r >= len(floatdata):		#推测的右端点超出范围，说明声音分隔为2帧，需要缓存
			# print('1')
			audio_cache['byte_audio'] = audiodata[int(WIDTH*l):]
			audio_cache['last_audio'] = floatdata[l:]	#检测到声音的开始，记录到缓存
			audio_cache['endpoint'] = r - len(floatdata)		#结束点在下个帧里的位置
			# return
		elif l != 0 or r != 0:		#声音片断在当前帧内，正常处理，不缓存
			if mode[:-1] == "sample":
				save_audio(audiodata[math.floor(WIDTH*l):math.ceil(WIDTH*r)])#保存声音
				return

			process(floatdata[l:r])
		else:
			# print('None')
			pass

# 测试程序
if __name__ == '__main__':
	sr = RATE
	x = newaudio(2, sr, 5.0)
	diswave(x, sr)
