from PIL import Image
import numpy as np
#from matplotlib import pyplot as plt

import re
import math
import random

from rknn.api import RKNN


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Load tensorflow model
    print('--> Loading model')
    #rknn.load_darknet(model='./yolov3_416x416.cfg', weight="./yolov3.weights")
    rknn.load_darknet(model='../cfg/yolov3_416x416.cfg', weight="../logs/cc.weights")
    print('done')

    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2', batch_size=1)

    # Build model
    print('--> Building model')
    rknn.build(do_quantization=True, dataset='./dataset.txt', pre_compile=True)
    print('done')

    #rknn.export_rknn('./yolov3_416x416.rknn')
    rknn.export_rknn("./new.rknn")
    exit(0)
