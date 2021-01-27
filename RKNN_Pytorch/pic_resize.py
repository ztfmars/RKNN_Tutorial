import cv2
import os



pic_dir = '/data/qualify/'
pic_size = (64, 64)






def pic_resize(pic_dir, pic_size):
    pic_list = os.listdir(pic_dir)
    for pic in pic_list:
        print("pic", pic)
        img = cv2.imread(pic_dir+pic)
        new_img = cv2.resize(img, (pic_size[0], pic_size[1]))
        cv2.imwrite(pic_dir+pic, new_img)









if __name__ == '__main__':
    pic_resize(pic_dir, pic_size)
