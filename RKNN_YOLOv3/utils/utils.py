"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import os, re, sys
import configparser

CONFIG_FILE = "cfg/global_config.cfg"


def read_cfg(CONFIG_FILE="cfg/global_config.cfg"):
    print("[INFO]cfg path:", os.path.join(os.getcwd(), CONFIG_FILE))
    if os.path.exists(os.path.join(os.getcwd(), CONFIG_FILE)):
        config_dict = {}
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        # 第一个参数指定要读取的段名，第二个是要读取的选项名
        config_dict["pretained_model"] = config.get("train_setting", "pretained_model")
        config_dict["anchors_path"] = config.get("train_setting", "anchors_path")
        config_dict["classes_path"] = config.get("train_setting", "classes_path")
        config_dict["annotation_path"] = config.get("train_setting", "annotation_path")
        config_dict["model_image_size"] = config.get("train_setting", "model_image_size")
        config_dict["Freeze_epoch"] = config.get("train_setting", "Freeze_epoch")
        config_dict["Epoch"] = config.get("train_setting", "Epoch")

        config_dict["classes"] = config.get("transfer_setting", "classes")
        config_dict["save_rknn_name"] = config.get("transfer_setting", "save_rknn_name")
        config_dict["manual_qualify_img"] = config.get("transfer_setting", "manual_qualify_img")

        return config_dict
    else:
        print("[ERROR] cfg path is not existed!")
        return None

def replace_words(file, old_str, new_str,flag=1):
    # replace yolo部分
    if flag:
        with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
            for line in f1:
                f2.write(re.sub(old_str, new_str, line))
        os.remove(file)
        os.rename("%s.bak" % file, file)
    else:
        # replace yolo前的conv
        with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
            lines = f1.readlines(1000000)

            for i in range(len(lines)):
                if lines[i]=="[convolutional]\n" \
                    and lines[i+1]=="size=1\n" \
                    and lines[i+2]=="stride=1\n" \
                    and lines[i+3]=="pad=1\n":
                    lines[i+4]=new_str+"\n"
            f2.writelines(lines)
            os.remove(file)
            os.rename("%s.bak" % file, file)



def change_fc_setting(cfg_dir, classes=80, model_size=416):
    if int(model_size) not in [320,416,608]:
        print("[ERROR] model input size is not 320,416 or 608, it's not supported!")
        sys.exit(0)
    file = "yolov3_%sx%s.cfg"%(model_size,model_size)
    filters= 3*(5+int(classes))
    # 替换[yolo] classes 变成实际训练数据的数目N
    replace_words(cfg_dir+file, old_str="classes=.*", new_str="classes=%d"%(int(classes)))

    # 替换 [yolo] 前一个[convolutional] filters=3×（N+5）
    replace_words(cfg_dir+file, old_str="filters=.*", new_str="filters=%d" % (int(filters)), flag=0)

    return cfg_dir+file


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32) / 255

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


if __name__ == '__main__':
    change_fc_setting(cfg_dir="../cfg/", classes=80, model_size=416)
