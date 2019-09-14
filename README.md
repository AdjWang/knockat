- 程序结构说明 -
knock
	/OriSamples		古老的采样测试数据和分类测试程序
	/Samples 		新的采样测试数据和分类测试程序
		/Samples1~5	采样数据
		ffttest.py	fft测试程序
		rename.py	采样数据重命名程序
		sampletrain.py	采样数据分类程序
		knock.m			分类模型
		test1KHz.wav	1KHz测试音频
	record.py			录音、信号处理
	operation.py		得到敲击位置后的应用程序
	main.py				主程序，创建主进程调用record.py和operation.py
	audioio.py 			音频文件及数据的输入输出
	globalvar.py 		全局变量和常量
	test.py 			测试
	reference.txt		参考资料
	readme.txt			(本文件)
	