
# PC环境
`ubuntu1604 or newer version`
`torch==1.6` or `torch==1.2`
`rknn-toolkit==1.6.0` or ``rknn-toolkit==1.4.0``

# 环境安装
[在PC上搭建RKNN模拟环境](https://blog.csdn.net/weixin_42237113/article/details/107023216)

# 训练步骤
tips:

- 数据分合：可以使用`predata.py` 将`data/update`中的数据按比例分配至`data/train`, `data/valid`中，之后再将这两个文件夹移动至data/data/即可， 参见`python predata --help`

		usage: predata.py [-h] [--train TRAIN] [ --valid VALID]  [--numclass NUMCLASS]  [--mode MODE]
		optional arguments:
				-h, --help           show this help message and exit
				--train TRAIN        训练数据的比例(0-100),默认为70
				--valid VALID        测试数据的比例(0-100),默认为30
				--numclass NUMCLASS  数据分类数目,默认为3 (对应目录0,1,2)
				 --mode MODE          生成数据模式:0 生成train和test的数据 清除数据模式:1 清空train和test下的数据.

- 图片resize： 可以使用`pic_resize.py`修改参数即可。

1 将需要训练的多类别图片放置到data/data下或者指定文件夹，依照需求修改config.cfg中训练参数

    [Data]
    train_datapath=data/data -> 存放数据的文件夹位置，下面的文件夹为train/valid，分别存放训练和验证的图片
    qualify_datapath = dataset.txt -> dataset.txt用于存储量化的图片；dataset中的图片最好是resize之后的图片;此处需要手动填写
    
    [Model]
    ntypes = 3 -> 最终生成的图像分类的数目
    input_shape = [224,224] -> 输入图片的尺寸，即模型入口参数
    
    [Train]
    epochs = 100 -> 解封toplayer之后，训练次数
    batch_size = 5 -> 训练时候的batch size
    lr_patience = 10 -> epochs > lr_patience, 当loss不再变化, 降低lr.
    lr_ratio = 0.3 -> lr 减小的倍率
    earlystop_patience = 30 -> ealy stop 停止的次数,即经过如此多次之后、loss不下降停止的次数；如果 == 0， 则不开启earlystop
    python_env=/xxx/torch1.2/bin/python -> 安装了torch1.2和rknn-toolkit所在的环境的python路径
    
    [Convert]
    rknn_savename=final -> 最后rknn生成的名字
    ;默认不变，否则需要改动训练时候的图片处理内容, 即train.py line 39
    cmv='127.5 127.5 127.5 127.5' -> channel_mean_value
    rc='0 1 2' -> reorder_channel
    
    [Evaluate]
    only_infer_and_evaluate = False -> 如果选择True, 只是进行rknn测评和评估(默认已经生成好了rknn)
    img_path = data/qualify/space_shuttle_224.jpg -> 测评用的图片
    rknn_path = ckpt/final.rknn -> 测评时候加载的预训练模型

2 resize若干张需要量化的图片，图片保存至特定目录data/qualify，量化图片路径下入dataset.txt
3 source /xxx/pytorch/torch1.2/bin/activate， 激活转化个人环境
4 本例中使用mobilenet v2作为训练的模型，读者可以自行修改模型，例如resnet101 等等，修改
train.py中 line 66-96
5 将train, valid 数据分别放到data/data里面不同类别的文件夹中
6  运行 `python main.py` ：
        

```python
    如果only_infer_and_evaluate = False， 只进行模型训练，rknn转化，最后在save_model保存最后的rknn
    模型；
    如果only_infer_and_evaluate = True，只是利用已生成的rknn模型对设定图片进行分类识别，性能评估
```

 
