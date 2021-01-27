# -* - coding: UTF-8 -* -
import os
import configparser
import re
import os

CONFIG_FILE = "config.cfg"

"""
type_list: {'Train': ['epochs', 'batch_size', 'lr_patience', 'lr_ratio', 
                        'earlystop_patience', 'train_all'],
            'Model': ['ntype', 'input_shape'], 
            'Data': ['train_datapath', 'qualify_datapath'],
            'Convert':["rknn_savename", "cmv", "rc"],
            'Evaluate':["only_infer_and_evaluate", "img_path", "rknn_path"] }
"""


# -------------------------------------------------------------------------------------------
#	读取cfg文件参数
# -------------------------------------------------------------------------------------------
def read_config():
    config_dict = {}
    print("**" * 50)
    print("[INFO] Begin to read config.cfg.")
    print("**" * 50)

    if os.path.exists(os.path.join(os.getcwd(), CONFIG_FILE)):
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        type_name = config.sections()
        # print("typename:", type_name)
        data_name = config.options(type_name[0])  # Data
        model_name = config.options(type_name[1])  # Model
        train_name = config.options(type_name[2])  # Train
        convert_name = config.options(type_name[3])  # Convert
        evaluate_name = config.options(type_name[4])  # Evaluate
        type_list = {"Data": data_name, "Train": train_name,
                     "Model": model_name, "Convert":convert_name,
                     "Evaluate":evaluate_name}
        # print("type_list:", type_list)

        for one_type in type_list:
            for item in type_list[one_type]:
                # 第一个参数指定要读取的段名，第二个是要读取的选项名
                value = config.get(one_type, item)
                config_dict[item] = value

        print("config_dict:", config_dict)
        return config_dict

    else:
        print("[ERROR] There is no valid config.cfg!Please recheck again!")
        exit(0)


# -------------------------------------------------------------------------------------------
#	根据参数进行模型训练
# -------------------------------------------------------------------------------------------
def train_process(config_dict):
    argument_str = " --data=" + config_dict["train_datapath"] + \
                   " --ntypes=" + config_dict["ntypes"] + \
                   " --input_shape=" + config_dict["input_shape"] + \
                   " --epochs=" + config_dict["epochs"] + \
                   " --batch_size=" + config_dict["batch_size"] + \
                   " --lr_patience=" + config_dict["lr_patience"] + \
                   " --lr_ratio=" + config_dict["lr_ratio"] + \
                   " --earlystop_patience=" + config_dict["earlystop_patience"] + \
                   " --train_all_layer" + config_dict["train_all_layer"]
    print("**" * 50)
    print("[INFO] Begin to train with added argument paraments")
    print("**" * 50)

    print("[INFO] Read train parameters:", argument_str)
    os.system(str(config_dict["python_env"])+" train.py" + argument_str)


# -------------------------------------------------------------------------------------------
#	pt文件转换成rknn文件
# -------------------------------------------------------------------------------------------
def cvt_process(config_dict):
    argument_str = " --qualify_datapath=" + config_dict["qualify_datapath"] + \
                   " --ntypes=" + config_dict["ntypes"] + \
                   " --input_shape=" + config_dict["input_shape"] + \
                   " --rknn_savename=" + config_dict["rknn_savename"] + \
                   " --cmv=" + config_dict["cmv"] + \
                   " --rc=" + config_dict["rc"] + \
                   " --rknn_path=" + config_dict["rknn_path"] + \
                   " --img_path=" + config_dict["img_path"] + \
                   " --only_infer_and_evaluate="+ config_dict["only_infer_and_evaluate"]
    print("**" * 50)
    print("[INFO] Begin to convert pytorch model to RKNN model with added argument paraments")
    print("**" * 50)

    print("[INFO] Read convert parameters:", argument_str)
    os.system(str(config_dict["python_env"]) + " convert.py" + argument_str)


if __name__ == '__main__':
    config_dict = read_config()
    if config_dict["only_infer_and_evaluate"] == 'False':
        train_process(config_dict)
    else:
        cvt_process(config_dict)
