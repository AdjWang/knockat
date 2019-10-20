# 无聊的时候敲敲桌子玩  
knock at... 其实什么都能敲  
这是一个识别敲击位置的程序，可以视为一个简单的事件检测程序。  
在桌子上有几个点，用小锤子去敲，我想让计算机知道我敲的位置，来点互动~  
然后这个程序诞生了。  

## 实现方案  
1. 单声道录音  
2. 端点检测：短时自相关  
3. 特征提取：MFCC  
4. 分类器：SVM  
5. 输出图形界面：matplotlib(demo程序方案，后续更换)

## 目录结构  
knockat/  
|-- main.py                  主程序-运行此程序启动  
|-- main/  
|-- |-- audioio.py              音频文件读写  
|-- |-- dataprocess.py      音频数据处理  
|-- |-- dsp.py                    数字信号处理  
|-- |-- globalvar.py           全局变量  
|-- |-- operation.py           识别后的操作  
|-- |-- record.py                麦克风操作，录音  
|-- modules/                    模型文件-训练自动产生  
|-- |-- \*.m  
|-- references/            参考资料  
|-- |--  ...  
|-- samples/                程序主体模块存放  
|-- |-- samples\*/           采样-数字\*对应采样点  
|-- test/                   功能开发测试  
|-- train/                  数据训练  
|-- |-- sampletrain.py  
|-- README.md              说明文档  
某些文件可能没有，有的需要首次运行时生成，有的后续会逐步添加。不影响正常使用。

## 启动方式:  
在knockat目录下运行终端：  
```sh
python3 main.py [--mode]
```
首次使用需要使用--mode参数采集音频、训练，后续使用直接运行main.py即可。

## 示例:
### 采集数据(首次运行需要)
在桌子上画5个点，建议间距不小于15cm
找一个小锤子

```sh
python3 main.py --mode sample1
```
敲100下，如果识别到敲击声，命令窗口会有提示
按Ctrl+C终止程序

```sh
python3 main.py --mode sample2
```
敲100下，如果识别到敲击声，命令窗口会有提示
按Ctrl+C终止程序

```sh
python3 main.py --mode sample3
```
敲100下，如果识别到敲击声，命令窗口会有提示
按Ctrl+C终止程序

```sh
python3 main.py --mode sample4
```
敲100下，如果识别到敲击声，命令窗口会有提示
按Ctrl+C终止程序

```sh
python3 main.py --mode sample5
```
敲100下，如果识别到敲击声，命令窗口会有提示
按Ctrl+C终止程序

### 训练采样数据(首次运行需要)
```sh
python3 main.py --mode train
```
### 开始玩耍
```sh
python3 main.py
```
  
后续计划改进内容：  
1. 可变点数  
2. 抗噪  

(to be continued...)

