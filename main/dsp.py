#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
信号处理
"""

#导入库
import math
import numpy as np
import librosa
import pyworld as pw
from scipy.fftpack import fft, ifft, dct
from scipy import signal
from pyAudioAnalysis import audioFeatureExtraction
from multiprocessing import Pool						#进程并行
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#导入其他文件

#类
#常量
#全局变量
#函数
def asnumpy(audiodata):
	'''	数据格式转换为numpy数组
		audiodata:音频数据
		返回numpy格式音频数据
	'''
	data = audiodata
	import numpy
	if not isinstance(audiodata, numpy.ndarray):
		data = numpy.array(audiodata)
	return data

def rfft(audiodata, n_result):
	'''	返回模值的fft(对称数据的前一半)
		audiodata:音频数据
		n_result:返回数据的范围 第n点对应频率:f=n*fs/N
	'''
	data = asnumpy(audiodata)
	'''
	--------------------- 
	作者：赵至柔 
	来源：CSDN 
	原文：https://blog.csdn.net/qq_39516859/article/details/79766697 
	版权声明：本文为博主原创文章，转载请附上博文链接！
	'''
	N = len(data)
	fftdata = fft(data, n=N)                     #快速傅里叶变换

	absdata = abs(fftdata)/(N/2)           #归一化处理
	return absdata[0 : min(n_result, N//2)]  #由于对称性，只取一半区间


def feature(audiodata, sr, coeffs):
	''' 音频特征提取函数
		audiodata:音频数据 numpy格式
		sr:采样频率
	'''
	coeffs = np.concatenate([coeffs[:int(len(coeffs)/2)], np.zeros(len(audiodata)-1).astype('complex64'), coeffs[int(len(coeffs)/2):]])
	delay = int((len(coeffs)-1)/2)		#延迟(N-1)Ts/2  N为滤波器阶数 Ts为一个采样点时间
	F = ifft(fft(audiodata, len(coeffs)) * coeffs).astype('float32')[delay : len(audiodata)+delay]
	# F = rfft(audiodata, 500)		#FFT 观察发现信息集中在500之前，对应频率500*44100/10240=2153.3203125(Hz)
	# F = librosa.feature.spectral_centroid(audiodata, sr)[0]#光谱质心
	# F = librosa.feature.mfcc(audiodata, sr).ravel()
	# F = audioFeatureExtraction.stFeatureExtraction(audiodata, sr, 0.030*sr, 0.015*sr)[0].ravel();
	# F, t = pw.dio(audiodata.astype('double'), sr, f0_floor= 0.0, f0_ceil= 2000.0, channels_in_octave= 1, frame_period=pw.default_frame_period)
	return F#/max(F)

def corrfilter(audiodata):
	'''	用声音数据频域平均作为滤波器参数
	'''
	cplx = fft(audiodata)
	coef = np.mean(cplx, axis=0)
	return coef

def sprocess(audiodata, window, frame_move, frame_len, func):
	'''	短时xxx
		audiodata:音频数据
		window:窗函数
		frame_move:帧移
		frame_len:帧长
		func:xxx函数 如：	短时过零率：func=zcr
						短时能量：func=energy
						短时谱熵：func=sep
	'''
	assert frame_len <= len(audiodata), '帧长不能大于音频数据长度'
	assert frame_move <= frame_len, '帧移不能大于帧长'
	data = asnumpy(audiodata)
	p = 0
	frame_data = []
	''' 测试程序：矩形窗，帧移为帧长 '''
	while len(data[p:p+frame_len]) > 0:
		frame_data.append(data[p:p+frame_len])
		p += frame_move

	pool = Pool(4)
	res = np.array(pool.map(func, frame_data))
	pool.close()
	pool.join()
	return res

def test(audiodata):
	# return energy(audiodata)*zcr(audiodata)
	corr = np.correlate(audiodata, audiodata, 'full')	#自相关
	result = np.max(np.abs(corr))
	return result#*zcr(audiodata)
	# Ym = abs(fft(audiodata))**2	#FFT模的平方作为能量谱
	# mfcc = MFCC_Filter(fs=44100, n_fft=len(audiodata))
	# mfcc_filter = mfcc.make()
	# return mfcc.filtering(Ym, mfcc_filter)

def zcr(audiodata):
	'''	过零率
		返回float
	'''
	def sgn(n):
		return 1 if n >= 0 else -1
	data = asnumpy(audiodata)
	# diff = np.diff(data)						#不按照公式规范数据，直接计算差分
	diff = np.diff(list(map(sgn, data)))	#公式方法
	return 0.5*np.sum(np.abs(diff))

def dtw(ts_a, ts_b, d=lambda x,y: abs(x-y), mww=10000):
    """Computes dtw distance between two time series
    https://www.cnblogs.com/ningjing213/p/10502519.html
    
    Args:
        ts_a: time series a
        ts_b: time series b
        d: distance function
        mww: max warping window, int, optional (default = infinity)
        
    Returns:
        dtw distance
    """
    
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
        # for j in range(1, N):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window 
    return cost[-1, -1]

def energy(audiodata):
	'''	能量
	'''
	data = asnumpy(audiodata)
	return np.sum(np.abs(data))		#所有数据绝对值之和(感觉：不使用平方防止数据变小丢失精度)

def sep(audiodata):
	'''	谱熵
		*未测试
	'''
	data = asnumpy(audiodata)
	n_half = len(data)//2 		#FFT对称，取一半简化计算
	# def pdf(audiodata, i):
	# 	''' 每个频率分量的归一化谱概率密度(pdf)
	# 		audiodata:音频数据
	# 		fs:采样频率
	# 		fi:选取的频率分量
	# 	'''
	rfftdata_half = abs(fft(data)[0:n_half])**2	#FFT模的平方作为能量谱
	Ym_sum = np.sum(rfftdata_half)*2 		#能量总和
	Hm_half = 0
	for i in range(n_half):
		# i = min(i, n_half)	#采样定理f<fs/2
		# fi = i*fs/len(data)		#第i点频率=第i点*采样频率/FFT点数
		pdf = rfftdata_half[i]/Ym_sum	#第i点概率密度
		if not math.isclose(pdf, 0.0):		#不能对0取对数,lim(0*ln0)极限为0
			Hm_half += pdf * math.log(pdf, 2)
	return -Hm_half*2/math.log(len(data), 2)		#归一化

# def vad(audiodata, gate=0.001):
# 	'''	端点检测
# 		使用自相关最大值/过零率
# 		audiodata:音频数据
# 		gate:门限
# 	'''
# 	data = asnumpy(audiodata)
# 	zcr_data = zcr(data)
# 	if math.isclose(zcr_data, 0.0):
# 		return 1
# 	corr = np.correlate(data, data, 'full')	#自相关
# 	result = np.max(np.abs(corr))			#自相关最大值
# 	# print(result)
# 	return 0 if result < gate else 1

def _corr(d):
	corr = np.correlate(d, d, 'full')	#自相关
	return np.max(np.abs(corr))			#自相关最大值

def vad(audiodata, gate=0.3, signallen=6400):
	'''	端点检测
		audiodata:音频数据
		gate:门限
		signallen:信号长度
	'''
	data = asnumpy(audiodata)
	frame_corr = sprocess(data, 0, 300, 600, _corr)
	if max(frame_corr)-np.mean(frame_corr) > gate:	#检测到有敲击
		peakl, peakr = peak_cut(frame_corr, frame_corr.argmax(), lk=0.1, rk=0)#分帧自相关尖峰左右端点
		l = int(peakl/len(frame_corr)*len(data))	#信号实际左端点
		r = l + signallen							#peakr不能测准，所以根据信号长度判断右端点
		return l, r
	else:				#信号中没有检测到敲击
		return 0, 0

def peak_cut(data, p_peak, lk, rk):
	'''	data:数据序列
		p_peak:需要查找的尖峰位置
		lk:左侧允许下降斜率(>=0)
		rk:右侧允许下降斜率(<=0)
	'''
	assert 0 <= p_peak < len(data), 'p_peak不在数据范围内'
	assert lk >= 0, '斜率lk不能小于0'
	assert rk <= 0, '斜率rk不能大于0'
	l, r = p_peak, p_peak
	while l > 0:
		if (data[l] - data[l-1]) < lk:
			break
		l -= 1
	while r < len(data)-1:
		if (data[r+1] - data[r]) > rk:
			break
		r += 1
	return l, r

class MFCC_Filter():
	'''	MFCC特征参数提取（基于MATLAB和Python实现）
		https://www.e-learn.cn/content/python/647603
	'''
	def __init__(self, fs, n_fft, n_filter=24, low_freq_mel=0):
		'''	fs:采样频率
			n_filter:三角滤波器个数
			low_freq_mel:滤波器低通频率
			n_fft:FFT点数
		'''
		self.fs = fs
		self.n_filter = n_filter
		self.low_freq_mel = low_freq_mel
		self.n_fft = n_fft

	def make(self):
		'''	生成滤波器
		'''
		#梅尔滤波器系数
		high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # 把 Hz 变成 Mel
		mel_points = np.linspace(self.low_freq_mel, high_freq_mel, self.n_filter + 2)  # 将梅尔刻度等间隔
		hz_points = (700 * (10**(mel_points / 2595) - 1))  # 把 Mel 变成 Hz
		bin = np.floor((self.n_fft + 1) * hz_points / self.fs)
		fbank = np.zeros((self.n_filter, int(np.floor(self.n_fft / 2 + 1))))
		for m in range(1, self.n_filter + 1):
			f_m_minus = int(bin[m - 1])   # left
			f_m = int(bin[m])             # center
			f_m_plus = int(bin[m + 1])    # right
			for k in range(f_m_minus, f_m):
				fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
			for k in range(f_m, f_m_plus):
				fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
		return fbank

	def filtering(self, Ym, mfccfilter):
		'''	滤波，计算MFCC倒谱系数
			Ym:能量谱(FFT模的平方作为能量谱)
			mfccfilter:MFCC滤波器
		'''
		filter_banks = np.dot(Ym[0:mfccfilter.shape[1]], mfccfilter.T)
		filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 数值稳定性
		filter_banks = 10 * np.log10(filter_banks)  # dB 
		filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
		#print(filter_banks)
		#DCT系数
		num_ceps = 12
		c2 = dct(filter_banks, type=2, axis=-1, norm='ortho')[ 1 : (num_ceps + 1)] # Keep 2-13
		#归一化倒谱提升窗口
		# lifts = [(1 + 6 * np.sin(np.pi * n / num_ceps)) for n in range(1, num_ceps+1)]
		lifts = [(1 + num_ceps/2 * np.sin(np.pi * n / num_ceps)) for n in range(1, num_ceps+1)]
		#print(lifts)   
		c2 *= lifts
		return np.diff(c2)

	def save(self, filename, mfccfilter):
		'''	保存滤波器
		'''
		np.savetxt(filename, mfccfilter)

	def load(self, filename):
		'''	读取滤波器
		'''
		return np.loadtxt(filename)


class FIR_Filter():
	def __init__(self, fs, omegp, omegs, deltp=1.002, delts=0.003):
		self.fs = fs
		self.omegp = omegp
		self.omegs = omegs
		self.deltp = deltp
		self.delts = delts
		self.N = self.n_estimate(fs, omegp, omegs, deltp, delts)
		self.coeffs = self.hn(self.N, [0.0, omegp, omegs, fs/2], [1, 0], Hz=fs, _type='bandpass')

	def n_estimate(self, fs, omegp, omegs, deltp=1.002, delts=0.003):
		'''	估算滤波器长度
			delp:通带偏差，即滤波器通带内偏离单位增益的最大值
			dels:阻带偏差，即滤波器阻带内偏离零增益的最大值
			omegp:通带边沿频率，即滤波器增益为1-deltp时对应的频率(模拟频率，单位Hz)
			omegs:阻带边沿频率，即滤波器增益为delts时对应的频率(模拟频率，单位Hz)
			fs:采样频率

			Kaiser公式:

					-20log[(δs*δp)^0.5] - 13
			N = ———————————————————————————————— + 1
					2.32*(ωse - ωpe)

			数字频率变换：
			ω = 2πf/fs
		'''
		assert deltp > 1.0, '滤波器通带内偏离单位增益的最大值必须大于1'
		assert delts > 0.0, '滤波器阻带内偏离零增益的最大值必须大于0'
		assert omegs > omegp, '阻带边沿频率必须大于通带边沿频率'

		fN = (-20*math.log((delts*(deltp-1))**0.5, 10)-13)	\
				/(2.32*2*math.pi*(omegs-omegp)/fs) + 1
		return math.ceil(fN)	#向上取整

	def hn(self, numtaps, bands, desired, weight=None, Hz=1, _type='bandpass', maxiter=25, grid_density=16):
		# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.remez.html
		bpass = signal.remez(numtaps, bands, desired, weight, Hz, _type, maxiter, grid_density)
		return bpass

	def show_freqz(self):
		freq, response = signal.freqz(self.coeffs)
		ampl = np.abs(response)

		import matplotlib.pyplot as plt
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.semilogy(freq/(2*np.pi), ampl, 'b-')  # freq in Hz
		plt.show()

	def filtering(self, x):
		res = signal.convolve(x, self.coeffs, 'full')
		delay = int((self.N-1)/2)		#延迟(N-1)Ts/2  N为滤波器阶数 Ts为一个采样点时间
		return res[delay : len(x)+delay]

#测试程序
if __name__ == '__main__':
	fir = FIR_Filter(44100, 200, 230, deltp=10**(0.02/20), delts=10**(-50/20))
	# N = fir.n_estimate(44100, 180, 230, deltp=10**(0.02/20), delts=10**(-50/20))
	# N = fir.n_estimate()
	# print(N)
	# bpass = signal.remez(N, [0.0, 180, 230, 22050], [1, 0], Hz=44100, type='bandpass')
	# freq, response = signal.freqz(bpass)
	# ampl = np.abs(response)

	# import matplotlib.pyplot as plt
	# fig = plt.figure()
	# ax1 = fig.add_subplot(111)
	# ax1.semilogy(freq/(2*np.pi), ampl, 'b-')  # freq in Hz
	# plt.show()

