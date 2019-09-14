#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio

CHUNK = 1024	#每个CHUNK大小
CHANNELS = 1 	#声道数量
RATE = 44100	#采样率

#采样数据格式 http://people.csail.mit.edu/hubert/pyaudio/docs/index.html
FORMAT = pyaudio.paInt16	#16 bit int
WIDTH = 2		#数据宽度

CHUNK_NUM = 10		#每个frame中的CHUNK数目

GATE = 0.3		#端点检测门限值
SIGNALLEN = 6400	#声音长度

exitFlag = 0
