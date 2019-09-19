#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
部分全局变量

有部分全局变量在很多地方用到，不容易分类，所以集中到这里
"""

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

exitFlag = 0 		#程序退出标识，ctrl+c触发

# | normal  | samplen  | train   |
# | 正常处理 | 第n点采样 | 模型训练 |
MODE_ENUM = ("normal", "sample1", "sample2", "sample3", "sample4", "sample5", "train")  # 程序输入参数，设置运行模式
