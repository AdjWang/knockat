- 程序结构说明 -  
knockat/  
	main/  
		audioio.py          音频文件读写  
		dataprocess.py      音频数据处理  
		dsp.py              数字信号处理  
		globalvar.py        全局变量  
		main.py             主程序-运行此程序启动  
		operation.py        识别后的操作  
		record.py           麦克风操作，录音  
	modules/                模型文件-训练自动产生  
		*.m                 
	references/             参考资料  
		...  
	samples/                程序主体模块存放  
		samples*/           采样-数字*对应采样点  
	test/                   功能开发测试  
	train/                  数据训练  
	README.md               说明文档  
  
启动方式：  
python3 ./main/main.py  

to be continued...  