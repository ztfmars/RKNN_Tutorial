## YOLOV3：You Only Look Once目标检测模型在Keras当中的实现
---

### 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

### 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| COCO-Train2017 | [yolo_weights.h5](https://github.com/bubbliiiing/yolo3-keras/releases/download/v1.0/yolo_weights.h5) | COCO-Val2017 | 416x416 | 38.1 | 66.8

### 所需环境(PC)
ubuntu 1604LTS or above 
tensorflow-gpu==1.14.0  
keras==2.3.1 
rknn-toolkit >=1.4

有关于如何搭建ubuntu环境和rknn模拟环境，参见

[ubuntu - blog](https://blog.csdn.net/weixin_42237113/article/details/107015030)

[simulator- blog](https://blog.csdn.net/weixin_42237113/article/details/107023216)

### 文件下载
训练所需的yolo_weights.h5可以去百度网盘下载  
链接: https://pan.baidu.com/s/1izPebZ6PVU25q1we1UgSGQ 提取码: tbj3  

### 预测步骤
#### 1、使用预训练权重
a、下载完库后解压，在百度网盘下载yolo_weights.h5，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
可完成预测。  
b、利用video.py可进行摄像头检测。  

#### 2、使用自己训练的权重
##### 2.1、模型在GPU服务器上测试
a、按照训练步骤训练。  
b、在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path": 'model_data/yolo_weights.h5',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt,
    "score" : 0.5,
    "iou" : 0.3,
    # 显存比较小可以使用416x416
    # 显存比较大可以使用608x608
    "model_image_size" : (416, 416)
}

```
c、运行predict.py，输入  
```python
img/street.jpg
```
可完成预测。  
d、利用video.py可进行摄像头检测。  
##### 2.2、模型在rk3399pro上测试
1.在test.py文件中，需要改动如下
- 修改一下CLASS变成你的类别名称，
- RKNN_MODEL_PATH改成你最终生成的.rknn文件，
- INPUT_SIZE按照你自己训练数据定义（416,320,608），
- im_file是你需要识别图片的路径

2 运行test.py即可

### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2yolo3.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   

```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7、**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：   

```python
classes_path = 'model_data/my_classes.txt'    
```
model_data/my_classes.txt文件内容为：   
```python
cat
dog
...
```
8、修改cfg/global_config.cfg配置相关训练参数
```python
[train_setting]
pretained_model=model_data/best_weight_711.h5
anchors_path=model_data/yolo_anchors.txt
classes_path=model_data/my_classes.txt
annotation_path=2007_train.txt
model_image_size=416
Freeze_epoch=20
Epoch=100


[transfer_setting]
classes=4
save_rknn_name=final4
manual_qualify_img=False
```
相关词语的表达的含义如下：

	pretained_model ： 加载预训练的模型，默认应该是yolo_weights.h5，参加上面下载地址
	anchors_path ： 锚点的位置，不用修改
	classes_path : 自己的分类名称，注意要和VOC中标注的名称相对应
	annotation_path ： 最终生成的训练数据的信息
	model_image_size ：只有414， 320， 608这3种类型
	Freeze_epoch ：进行fine-turning，先封住主干训练toplayer,这里是训练次数
	Epoch ： 解封之后的训练次数
	classes ：训练的目标的分类数量
	save_rknn_name ：最后转化成的rknn模型的名称，不用加rknn，只需要名称即可
	manual_qualify_img ：默认进行量化的图片，选择false则自动选择一张；选择True的话，需要在transfer/dataset.txt中写入量化图片的具体全路径（最好是能resize到和model_image_size大小一致）

9、运行train.py即可开始训练。
如果转化过程中有问题的话，可以在train.py注释掉`train_process(config_dic)`,配置好参数只进行模型转化（h5->darknet->rknn）。

### mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

### Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP
