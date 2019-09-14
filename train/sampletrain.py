#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool						#进程并行
# from multiprocessing.dummy import Pool as ThreadPool	#线程并发
from sklearn import svm, model_selection, preprocessing
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import librosa
from pyAudioAnalysis import audioFeatureExtraction
from functools import partial

sys.path.append('../main/')
import dsp 			#../dsp.py
import audioio		#../audioio.py

#5个点的采样数据目录
SAMPLEPATH_P1 = '../samples/Samples1'
SAMPLEPATH_P2 = '../samples/Samples2'
SAMPLEPATH_P3 = '../samples/Samples3'
SAMPLEPATH_P4 = '../samples/Samples4'
SAMPLEPATH_P5 = '../samples/Samples5'
FILETYPE = '.wav'				#数据文件格式，其余文件不识别


def readmodel(filename):
	''' 读取训练模型 '''	
	return joblib.load(filename)

def savemodel(model, filename):
	''' 保存训练模型 '''
	joblib.dump(model, filename)

if __name__ == '__main__':
	print('读取数据文件列表...', end='', flush=True)
	datafilelist1 = audioio.loaddatafile(SAMPLEPATH_P1, FILETYPE)
	datafilelist2 = audioio.loaddatafile(SAMPLEPATH_P2, FILETYPE)
	datafilelist3 = audioio.loaddatafile(SAMPLEPATH_P3, FILETYPE)
	datafilelist4 = audioio.loaddatafile(SAMPLEPATH_P4, FILETYPE)
	datafilelist5 = audioio.loaddatafile(SAMPLEPATH_P5, FILETYPE)
	datafilelist = datafilelist1+datafilelist2+datafilelist3+datafilelist4+datafilelist5
	# datafilelist = audioio.loaddatafile('./', FILETYPE) + datafilelist1 + datafilelist2 + datafilelist4
	# datafilelist.pop(0)
	print('ok')
	# print(datafilelist)

	print('加载数据...', end='', flush=True)
	data = audioio.loaddata(datafilelist)	#函数内部已经格式转换
	# audiodata = np.array([i[0] for i in data])
	for i in range(len(data)):
		if data[i].shape[0] < 6400:
			data[i] = np.concatenate([data[i], np.zeros(6400-data[i].shape[0], dtype='float32')])
	audiodata = np.array(data)
	print('ok')

	# processdata = np.array([audiodata[0][:37000], (audiodata[0][37000:43400]+audiodata[1][8000:14400]), audiodata[0][43400:]])
	# processdata = np.concatenate(processdata)
	# processdata = audiodata[0]
	# noise = np.random.normal(0, 0.03, (len(processdata), 1)).ravel()
	# processdata += noise

	# fir = dsp.FIR_Filter(44100, 200, 230, deltp=1.002, delts=0.005)
	# x = audiodata[0]
	# res = fir.filtering(x)

	# xticks = [i*44100/6400 for i in range(500)]
	# plt.subplot(2,2,1)
	# plt.plot(audiodata[0])
	# plt.subplot(2,2,2)
	# plt.plot(xticks, dsp.rfft(audiodata[0], 500))
	# plt.subplot(2,2,3)
	# plt.plot(res)
	# plt.subplot(2,2,4)
	# plt.plot(xticks, dsp.rfft(res, 500))
	# plt.show()

	print('特征提取...', end='', flush=True)
	n_fft = 512
	mfcc = dsp.MFCC_Filter(44100, n_fft)
	mfcc_filter = mfcc.make()

	fir = dsp.FIR_Filter(44100, 200, 330, deltp=1.008, delts=0.005)
	print(fir.N)
	# coeffs = np.load('coef.npy')
	pool = Pool(4)
	featuredata = np.array(pool.map(fir.filtering, audiodata))					#fir
	ym = np.array(pool.map(partial(dsp.rfft, n_result=n_fft), audiodata))**2 		#能量谱
	featuredata = np.array(pool.map(partial(mfcc.filtering, mfccfilter=mfcc_filter), ym))	#mfcc
	pool.close()
	pool.join()
	print('ok')

	y = np.array(range(5)).repeat(100)#np.array(range(data.shape[0])).repeat(data.shape[1])

	# print('数据降维...', end='', flush=True)
	# pca = PCA(n_components=11)
	# reduceddata = pca.fit_transform(featuredata)
	# reduceddata = reduceddata/np.max(np.abs(reduceddata))		#normalize
	# savemodel(pca, '../modules/pca.m')	#保存模型

	# lda = LDA(n_components=4)
	# reduceddata = lda.fit_transform(featuredata, y)
	# reduceddata = reduceddata/np.max(np.abs(reduceddata))		#normalize
	# savemodel(lda, '../modules/lda.m')	#保存模型
	# print('ok')

	# plt.subplot(5,1,1)
	# # plt.plot(np.arange(500).T*44100/6400, np.sum(featuredata[0:100,:], axis=0))
	# plt.plot(np.arange(500).T*44100/6400, featuredata[0:100,:].T)

	# plt.subplot(5,1,2)
	# # plt.plot(np.sum(featuredata[100:200,:], axis=0))
	# plt.plot(np.arange(500).T*44100/6400, featuredata[100:200,:].T)

	# plt.subplot(5,1,3)
	# # plt.plot(np.sum(featuredata[200:300,:], axis=0))
	# plt.plot(np.arange(500).T*44100/6400, featuredata[200:300,:].T)

	# plt.subplot(5,1,4)
	# # plt.plot(np.sum(featuredata[300:400,:], axis=0))
	# plt.plot(np.arange(500).T*44100/6400, featuredata[300:400,:].T)
	
	# plt.subplot(5,1,5)
	# # plt.plot(np.sum(featuredata[400:500,:], axis=0))
	# plt.plot(np.arange(500).T*44100/6400, featuredata[400:500,:].T)
	# plt.show()

	x_train, x_test, y_train, y_test = model_selection.train_test_split(featuredata, y, random_state=0, test_size=0.3)#随机划分数据
	print('训练数据集：' + str(x_train.shape[0]))
	print('测试数据集：' + str(x_test.shape[0]))

	print('训练模型...', end='', flush=True)
	#'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
	classifier = svm.SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr', C=0.5, probability=True)
	# classifier = LDA(n_components=4)
	classifier.fit(x_train, y_train)	#测试训练
	# classifier.fit(reduceddata, y)		#全数据训练
	# savemodel(classifier, '../modules/svm.m')	#保存模型
	# classifier = svm_readmodel('../modules/knock.m')		#读取模型
	print('ok')

	print("输出训练集的准确率为：", classifier.score(x_train, y_train))
	print("输出测试集的准确率为：", classifier.score(x_test, y_test))


