import subprocess
import cv2
import numpy as np

import os
from matplotlib import pyplot as plt
from collections import Counter

import numpy as np
import cv2


def compute_mean(img):
    per_image_rmean = []
    per_image_gmean = []
    per_image_bmean = []
    per_image_bmean.append(np.mean(img[:, :, 0]))
    per_image_gmean.append(np.mean(img[:, :, 1]))
    per_image_rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_rmean)
    G_mean = np.mean(per_image_gmean)
    B_mean = np.mean(per_image_bmean)
    return R_mean, G_mean, B_mean


def pull_screenshot(img_path, SCREENSHOT_WAY):
    if 1 <= SCREENSHOT_WAY <= 3:
        process = subprocess.Popen(
            'adb shell screencap -p',
            shell=True, stdout=subprocess.PIPE)
        binary_screenshot = process.stdout.read()
        if SCREENSHOT_WAY == 2:
            binary_screenshot = binary_screenshot.replace(b'\r\n', b'\n')
        elif SCREENSHOT_WAY == 1:
            binary_screenshot = binary_screenshot.replace(b'\r\r\n', b'\n')
        lenb = len(binary_screenshot)
        if lenb > 0:
            f = open(img_path, 'wb')
            f.write(binary_screenshot)
            f.close()
            return img_path
        else:
            print("error! no screenshot data!")
    elif SCREENSHOT_WAY == 0:
        screenshot_name = "screenshot.png"
        os.system('adb shell screencap -p /sdcard/{}'.format(screenshot_name))
        os.system('adb pull /sdcard/{} {}'.format(screenshot_name, img_path))


def is_dark(img, pic_path, pic):
    # 把图片转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取灰度图矩阵的行数和列数
    r, c = gray_img.shape[:2];
    dark_sum = 0  # 偏暗的像素 初始化为0个
    dark_prop = 0  # 偏暗像素所占比例初始化为0
    piexs_sum = r * c  # 整个弧度图的像素个数为r*c

    # 遍历灰度图的所有像素
    for row in gray_img:
        for colum in row:
            if colum < 40:  # 人为设置的超参数,表示0~39的灰度值为暗
                dark_sum += 1
    dark_prop = dark_sum / (piexs_sum)
    print("dark_sum:" + str(dark_sum))
    print("piexs_sum:" + str(piexs_sum))
    print("dark_prop=dark_sum/piexs_sum:" + str(dark_prop))
    if dark_prop >= 0.6:  # 人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
        print(pic_path + " is dark!")
        return True
    else:
        print(pic_path + " is bright!")
        return False
    # hist(pic_path);  #若要查看图片的灰度值分布情况,可以这个注释解除


crop_x, crop_y, crop_w, crop_h = (60 * 2, 179 * 2, 850, 850)  # 手动定位出色块区域


def parse_color(file_name):
    img = cv2.imread(file_name)
    # x, y, w, h = (60 * 2, 179 * 2, 850, 850)
    crop = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    binary_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if is_dark(crop, file_name, ""):
        ret, binary_img = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    else:
        ret, binary_img = cv2.threshold(binary_img, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # plt.imshow(binary_img)
    # plt.show()

    # 寻找轮廓
    h = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 提取轮廓
    contours = h[0]
    # 打印返回值，这是一个元组
    # print(type(h))
    # 打印轮廓类型，这是个列表
    # print(type(h[1]))
    # 查看轮廓数量
    print(len(contours))

    # 画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
    # cv2.drawContours(crop, contours, -1, (0, 255, 0), 1)
    fixed = 5
    mydict = {}
    mean_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        _crop = crop[y + fixed:y + h - fixed, x + fixed:x + w - fixed]
        cur_mean = compute_mean(_crop)  # 计算每个小色块的颜色平均值
        mydict[cur_mean] = (x, y, w, h)  # 平均值做key，记录轮廓坐标，最后只会剩下两个值
        mean_list.append(cur_mean)  # 用于计算两个值的个数,个数为1的就是不同的
    print(mydict)
    # print(mean_list)
    count = Counter(mean_list)
    print(count)
    for key in count:
        if count[key] == 1:
            print(mydict[key])
            return mydict[key]
    #         cv2.rectangle(crop, (mydict[key][0], mydict[key][1]),
    #                       (mydict[key][0] + mydict[key][2], mydict[key][1] + mydict[key][3]), (255, 255, 255), 3)
    # plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    # plt.show()


def touch_pos(pos_x, pos_y):
    os.system('adb shell input tap ' + pos_x + " " + pos_y)


(x, y, w, h) = parse_color(pull_screenshot("./shot/1.png", 2))

if x and y and w and h:
    touch_pos(str(x + crop_x + 10), str(y + crop_y + 10))
# parse_color("./test1.png")
# print(x + crop_x, y + crop_y, w, h)
# img = cv2.imread("./shot/1.png")
# cv2.circle(img, (x + crop_x + 10, y + crop_y + 10), 2, (255, 255, 255), 3)

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
