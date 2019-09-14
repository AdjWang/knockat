#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
# import webrtcvad
import librosa
from librosa import display
from scipy.fftpack import fft
from sklearn.externals import joblib

import audioio as io
import dsp
from globalvar import *		#导入常量


def readmodel(filename):
	return joblib.load(filename)

def diswave(x, sr, ion=False):
	'''	显示波形幅度包络图
		x:波形数据 sr:采样频率 ion:pyplot显示模式，默认阻塞模式
	'''
	if ion:
		plt.ion()
		plt.clf()

	display.waveplot(x, sr)

	if ion:
		plt.pause(0.01)
	if not ion:
		plt.show()

audio_cache = {}	#记录声音数据，用于连续录音
pca = readmodel('../modules/pca.m')
classifier = readmodel('../modules/svm.m')		#读取模型
# coef = np.load('../modules/coef.npy')
fir = dsp.FIR_Filter(44100, 300, 330, deltp=1.008, delts=0.005)
def dataprocess(audiodata, callback = None):
	'''	数据处理
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
	'''
	def process(floatdata):
		# floatdata, sr = librosa.load('./test.wav')	#读取声音

		diswave(floatdata, RATE, ion=True)		#绘制波形
		
		# featuredata = dsp.feature(floatdata, RATE)					#FFT
		featuredata = fir.filtering(floatdata)
		reduceddata = pca.transform(featuredata.reshape(1, -1))

		probability = classifier.predict_proba(reduceddata)	#分类
		position = np.argmax(probability)					#概率最大的位置
		print(probability)
		print(position)

		# print(position, max(probability))
		if callback:				#触发
			if callable(callback):		#是一个函数
				callback(position)
			elif isinstance(callback, multiprocessing.queues.Queue):	#进程通信队列
				callback.put(position, block=False)


	tupdata = struct.unpack('<'+len(audiodata)//WIDTH*'h', audiodata)#音频数据byte转int
	floatdata = np.array(tupdata, dtype='float32')/float(1 << ((8 * WIDTH) - 1))		#归一化数据

	if len(audio_cache) > 0:	#有缓存，需要连接剩下的声音
		# io.saveaudio('./test.wav', audio_cache['byte_audio']+audiodata[:math.ceil(WIDTH*audio_cache['endpoint'])], CHANNELS, FORMAT, RATE)#保存声音
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
			# io.saveaudio('./test.wav', audiodata[math.floor(WIDTH*l):math.ceil(WIDTH*r)], CHANNELS, FORMAT, RATE)#保存声音
			process(floatdata[l:r])
		else:
			# print('None')
			pass

if __name__ == '__main__':
	sr = RATE
	x = newaudio(2, sr, 5.0)
	diswave(x, sr)
