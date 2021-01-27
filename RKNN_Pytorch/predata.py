"""
训练数据的分发处理
把采集的数据data/update下的图片,按照一定比例复制到训练目录data/train和验证目录data/valid里
"""

import argparse
import random
import os
import shutil

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train",
    default='70',
    help="训练数据的比例(0-100),默认为70")

parser.add_argument(
    "--valid",
    default='30',
    help="测试数据的比例(0-100),默认为30")

parser.add_argument(
    "--numclass",
    default=3,
    help="数据分类数目,默认为3 (对应目录0,1,2)")

parser.add_argument(
    "--mode",
    default='0',
    help="生成数据模式:0 生成train和test的数据   清除数据模式:1 清空train和test下的数据.")

args = parser.parse_args()

VALID_EXTENSIONS = ('png', 'jpg')


def getfilelist(path):
    import os
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    for file in files:  # 遍历文件夹
        if file.lower().endswith(VALID_EXTENSIONS):
            s.append(path + '/' + file)  # 每个文件的文本存到list中
    s.sort()
    return s


'''
数据分割算法
源数据目录         path0 =['./data/updata/0', './data/updata/1', './data/updata/2']
目标数据目录(训练)  path1 =['./data/train/0', './data/train/1', './data/train/2']
目标数据目录(测试)  path2 =['./data/test/0', './data/test/1', './data/test/2']
copydata(path0, path1, path2)
'''


def copydata(path0, path1, path2):
    for i in range(int(args.numclass)):
        # 先分割数据集
        r = getfilelist(path0[i])
        random.shuffle(r)
        mcount = len(r)
        train0 = int(args.train)
        test0 = int(args.test)
        count0 = int(mcount * train0 / (train0 + valid0))
        r0 = r[:count0]
        r1 = r[count0:]
        print('DIR:%s,Total:%s,Train:%s,valid:%s' % (path0[i], len(r), len(r0), len(r1)))

        # 再按照分割好的复制数据
        for i0 in r0:
            dst = path1[i] + '/' + os.path.basename(i0)
            shutil.copyfile(i0, dst)
        for i1 in r1:
            dst = path2[i] + '/' + os.path.basename(i1)
            shutil.copyfile(i1, dst)


if __name__ == '__main__':
    
    path00 = ['./data/update/0', './data/update/1', './data/update/2', './data/update/3','./data/update/4', './data/update/5', './data/update/6', './data/update/7','./data/update/8', './data/update/9']
    path10 = ['./data/train/0', './data/train/1', './data/train/2', './data/train/3','./data/train/4', './data/train/5', './data/train/6', './data/train/7','./data/train/8', './data/train/9']
    path20 = ['./data/valid/0', './data/valid/1', './data/valid/2', './data/valid/3','./data/valid/4', './data/valid/5', './data/valid/6', './data/valid/7','./data/valid/8', './data/valid/9']

    numclass = int(args.numclass)
    path0 = path00[:numclass]
    path1 = path10[:numclass]
    path2 = path20[:numclass]

    if args.mode == '0':
        print('start copydata to train & valid dir...')
        copydata(path0, path1, path2)
        print('OK, work end.')

    if args.mode == '1':
        print('start to remove dir and rebuild new dir...')
        for i in path1:
            try:
                shutil.rmtree(i)
            except:
                pass
            os.makedirs(i)
        for i in path2:
            try:
                shutil.rmtree(i)
            except:
                pass
            os.makedirs(i)
        print('DIR is remove, new dir is maked. ')
