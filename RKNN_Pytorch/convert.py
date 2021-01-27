import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
import argparse
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------------------------
#	生成torchscript文件
# -------------------------------------------------------------------------------------------
def export_pytorch_model(model_load="./ckpt/final.pt", input_shape=[224, 224]):
    net = torch.load(model_load)
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1, 3, input_shape[0], input_shape[1]).to(device))
    # print("##"*10)
    # print(trace_model)
    # print("##" * 10)
    trace_model.save('./ckpt/tmp.pt')


def show_outputs(output, ntypes=3):
    output_sorted = sorted(output, reverse=True)
    top_str = '\n-----TOP %d-----\n' % ntypes
    for i in range(ntypes):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= ntypes:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top_str += topi
    print(top_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


# -------------------------------------------------------------------------------------------
#	pt文件转化成rknn文件
# -------------------------------------------------------------------------------------------
def conver_to_rknn(cmv='127.5 127.5 127.5 127.5',
                   rc='0 1 2',
                   pt_model='./ckpt/tmp.pt',
                   input_shape=[224, 224],
                   dq=True,
                   dt='dataset.txt',
                   save_name='final'):
    print("[INFO]Begin to convert pytorch model to rknn model!")
    # Create RKNN object
    rknn = RKNN()

    # pre-process config : [-1,1], RGB(0 1 2)
    print('---------------------> Config model')
    rknn.config(channel_mean_value=cmv, reorder_channel=rc)
    print('done')

    # Load pytorch model
    input_shape.insert(0, 3)
    input_size_list = [input_shape]
    print('---------------------> Loading model')
    ret = rknn.load_pytorch(model=pt_model, input_size_list=input_size_list)
    if ret != 0:
        print('Load pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('---------------------> Building model')
    ret = rknn.build(do_quantization=dq, dataset=dt)
    if ret != 0:
        print('Build pytorch failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('---------------------> Export RKNN model')
    export_rknn_path = "save_model/"+save_name+".rknn"
    ret = rknn.export_rknn(export_rknn_path)
    if ret != 0:
        print('Export mbv2.rknn failed!')
        exit(ret)
    print('done')


# -------------------------------------------------------------------------------------------
#	利用生成rknn文件进行单张图片的推理、模型性能评估
# -------------------------------------------------------------------------------------------
def inference_and_evaluate_model(rknn_path='./save_model/mbv2.rknn',
                                 img_path='./data/data/qualify/space_shuttle_224.jpg',
                                 ntypes=3,
                                 input_shape=[224,224]):
    print("[INFO]Begin to load rknn model, inference and evaluate the model!")
    # Create RKNN object
    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print('Load model failed')
        exit(ret)

    # Set inputs
    img = cv2.imread(img_path)
    img = cv2.resize(img, (input_shape[0],input_shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('---------------------> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('---------------------> Running model')
    outputs = rknn.inference(inputs=[img])
    show_outputs(softmax(np.array(outputs[0][0])), ntypes=ntypes)
    print('done')

    # perf
    print('---------------------> Begin evaluate model performance')
    perf_results = rknn.eval_perf(inputs=[img])
    print(perf_results)
    print('done')

    rknn.release()


# -------------------------------------------------------------------------------------------
#	转换主函数
# -------------------------------------------------------------------------------------------
def convert_process(args):
    dataset = args.qualify_datapath

    ntypes = int(args.ntypes)
    tmp = re.findall(r"\d+\.?\d*", args.input_shape)
    input_shape = [int(i) for i in tmp]

    save_name = args.rknn_savename
    cmv = args.cmv
    rc = args.rc

    rknn_path = args.rknn_path
    img_path = args.img_path
    infer_and_evaluate = args.only_infer_and_evaluate

    if infer_and_evaluate == 'True':
        inference_and_evaluate_model(rknn_path=rknn_path, img_path=img_path,
                                     input_shape=input_shape, ntypes=ntypes)
    else:
        export_pytorch_model(model_load="./ckpt/final.pt", input_shape=input_shape)
        conver_to_rknn(input_shape=input_shape, dt=dataset,
                       cmv=cmv, rc=rc,
                       save_name=save_name)




if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser()
    # [data]
    parser.add_argument("--qualify_datapath", help="量化图片位置文件", default='dataset.txt')
    # [model]
    parser.add_argument("--ntypes", help="分类数目.", default=3)
    parser.add_argument("--input_shape", help="模型中的输入尺寸.(图片的resize尺寸)",
                        default='[224,224]')
    # [convert]
    parser.add_argument("--rknn_savename", help="rknn最终保存名称.", default="final")
    parser.add_argument("--cmv", help="r b g mean.", default='127.5 127.5 127.5 127.5')
    parser.add_argument("--rc", help="图片读取方式是rgb 还是bgr,默认rgb", default='0 1 2')
    # [evaluate]
    parser.add_argument("--only_infer_and_evaluate", help="是否进行单张测试图片的推理和模型性能评估.",
                        default='True')
    parser.add_argument("--rknn_path", help="评估的时候使用的已生成的rknn模型路径.",
                        default="save_model/final.rknn")
    parser.add_argument("--img_path", help="推理和评估所使用的单张测试图片",
                        default='./data/qualify/space_shuttle_224.jpg')
    args = parser.parse_args()
    convert_process(args)
