import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys
import os
import Store
import config
import tkinter as tk

configMain = config.config



def returnGaussianBlurImg(img):
    blur = configMain.blur
    img = cv2.GaussianBlur(img, (blur, blur), 0)
    Store.blurImg = img.copy()
    return img





def returnEdegImg(img):
    ret, imgBinary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgEdge = cv2.Canny(imgBinary, 100, 200)
    kernel = np.ones((configMain.morphologyr, configMain.morphologyc), np.uint8)
    img_edge1 = cv2.morphologyEx(imgEdge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    Store.imgEdge = img_edge2.copy()
    return img_edge2


def returnEdgePointList(eageList):
    contours, hierarchy = cv2.findContours(eageList, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def dropEdgePointByMinArea(edgePointList):
    edgePointList = [cnt for cnt in edgePointList if cv2.contourArea(cnt) > configMain.Min_Area]
    return edgePointList





def trunEdgePointToRectAndBoxPoint(edgePointList):
    rectangleList = []
    boxPointList = []
    for edgePoint in edgePointList:
        rect = cv2.minAreaRect(edgePoint)
        rectangleList.append(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxPointList.append(box)

    Store.boxPointList = boxPointList.copy()
    Store.rectangleList = rectangleList.copy()

    return rectangleList, boxPointList


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def returnCardImgList(rectangleList):
    card_imgs = []
    for rect in rectangleList:
        if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除

        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [Store.picWid, Store.picHei]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        if left_point[1] <= right_point[1]:  # 正角度
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)

            dst = cv2.warpAffine(Store.RGBImg, M, (Store.picWid, Store.picHei))
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            card_imgs.append(card_img)
        # cv2.imshow("card", card_img)
        # cv2.waitKey(0)
        elif left_point[1] > right_point[1]:  # 负角度

            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(Store.RGBImg, M, (Store.picWid, Store.picHei))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            card_imgs.append(card_img)

            # new_left_point = [left_point[0], heigth_point[1]]
            # pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            # pts1 = np.float32([left_point, heigth_point, right_point])
            # M = cv2.getAffineTransform(pts1, pts2)
            # dst = cv2.warpAffine(Store.RGBImg, M, (Store.picWid, Store.picHei))
            # point_limit(new_left_point)
            # point_limit(heigth_point)
            # point_limit(left_point)
            # card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            # card_imgs.append(card_img)
        # cv2.imshow("card", card_img)
        # cv2.waitKey(0)
    # Store.cardList=card_imgs.copy() 还需要颜色定位
    return card_imgs


def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    # col_num_limit = self.cfg["col_num_limit"]
    row_num_limit = configMain.row_num_limit
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


def dropCardImgByColorAndReturnColorList(card_imgs):
    newCard_imgs = []
    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        try:
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        except cv2.error:
            continue

        # 有转换失败的可能，原因来自于上面矫正矩形出错
        if card_img_hsv is None:
            continue
        row_num, col_num = card_img_hsv.shape[:2]
        card_img_count = row_num * col_num

        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 11 < H <= 34 and S > 34:  # 统计黄色点
                    yello += 1
                elif 35 < H <= 99 and S > 34:  # 统计绿色点
                    green += 1
                elif 99 < H <= 124 and S > 34:  # 统计蓝色点
                    blue += 1

                if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"

        limit1 = limit2 = 0
        if yello * 2 >= card_img_count:
            color = "yellow"
            limit1 = 11
            limit2 = 34  # 有的图片有色偏偏绿
        elif green * 2 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue * 2 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124  # 有的图片有色偏偏紫
        elif black + white >= card_img_count * 0.7:  # TODO
            color = "bw"

        if color != 'no':
            newCard_imgs.append(card_img)
            colors.append(color)
    Store.colors=colors.copy()



    return newCard_imgs, colors


def accurateCardPlaceByColor(card_imgs, colors):
    for card_index, card_img in enumerate(card_imgs):

        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        row_num, col_num = card_img_hsv.shape[:2]
        card_img_count = row_num * col_num
        limit1 = 0
        limit2 = 0
        if colors[card_index] == "yellow":
            limit1 = 11
            limit2 = 34  # 有的图片有色偏偏绿
        elif colors[card_index] == "green":
            limit1 = 35
            limit2 = 99
        elif colors[card_index] == "blue":
            limit1 = 100
            limit2 = 124  # 有的图片有色偏偏紫

        xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, colors[card_index])
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if colors[card_index] != "green" or yl < (
                yh - yl) // 4 else card_img[yl - (
                yh - yl) // 4:yh, xl:xr]
        if need_accurate:  # 可能x或y方向未缩小，需要再试一次
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, colors[card_index])
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if colors[card_index] != "green" or yl < (
                yh - yl) // 4 else card_img[yl - (
                yh - yl) // 4:yh, xl:xr]

    Store.card_imgs = card_imgs.copy()
    return card_imgs, colors


def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


def separateChar(card_imgs, colors):
    roi = None
    card_color = None
    charList = []

    for i, color in enumerate(colors):
        if color in ("blue", "yellow", "green"):
            card_img = card_imgs[i]
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
            if color == "green" or color == "yellow":
                gray_img = cv2.bitwise_not(gray_img)
            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 查找水平直方图波峰
            x_histogram = np.sum(gray_img, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / 2
            wave_peaks = find_waves(x_threshold, x_histogram)
            if len(wave_peaks) == 0:
                print("peak less 0:")
                continue
            # 认为水平方向，最大的波峰为车牌区域
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])

            gray_img = gray_img[wave[0]:wave[1]]
            # 查找垂直直方图波峰
            row_num, col_num = gray_img.shape[:2]
            # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
            gray_img = gray_img[1:row_num - 1]

            h, w = gray_img.shape[:2]
            if i == 0:
                gray_img = gray_img[:, 0 + 2:w]

            y_histogram = np.sum(gray_img, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

            wave_peaks = find_waves(y_threshold, y_histogram)

            # for wave in wave_peaks:
            #	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
            # 车牌字符数应大于6
            if len(wave_peaks) <= 6:
                print("peak less 1:", len(wave_peaks))
                continue

            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            max_wave_dis = wave[1] - wave[0]
            # 判断是否是左侧车牌边缘
            if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                wave_peaks.pop(0)

            # 组合分离汉字
            cur_dis = 0
            for i, wave in enumerate(wave_peaks):
                if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            if i > 0:
                wave = (wave_peaks[0][0], wave_peaks[i][1])
                wave_peaks = wave_peaks[i + 1:]
                wave_peaks.insert(0, wave)

            # 去除车牌上的分隔点
            point = wave_peaks[2]
            if point[1] - point[0] < max_wave_dis / 3:
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

            if len(wave_peaks) <= 6:
                print("peak less 2:", len(wave_peaks))
                continue
            part_cards = seperate_card(gray_img, wave_peaks)
            for i, part_card in enumerate(part_cards):
                # 可能是固定车牌的铆钉
                if np.mean(part_card) < 255 / 5:
                    print("a point")
                    continue
                part_card_old = part_card
                w = abs(part_card.shape[1] - 20) // 2

                part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                part_card = cv2.resize(part_card, (20, 20), interpolation=cv2.INTER_AREA)
                charList.append(part_card)

                # try:
                #     cv2.imshow('1', part_card)
                # except cv2.error:
                #     continue
                # cv2.waitKey(0)

                # part_card = deskew(part_card)
                # part_card = preprocess_hog([part_card])
                # if i == 0:
                #     resp = self.modelchinese.predict(part_card)
                #     charactor = provinces[int(resp[0]) - PROVINCE_START]
                # else:
                #     resp = self.model.predict(part_card)
                #     charactor = chr(resp[0])
                # # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                # if charactor == "1" and i == len(part_cards) - 1:
                #     if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                #         continue
                # predict_result.append(charactor)

            break

    Store.charList = charList.copy()
    return charList


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * 20 * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def train_svm(self):
    # 识别英文字母和数字
    self.model = SVM(C=1, gamma=0.5)
    # 识别中文
    self.modelchinese = SVM(C=1, gamma=0.5)
    if os.path.exists("svm.dat"):
        self.model.load("svm.dat")
    else:
        chars_train = []
        chars_label = []

        for root, dirs, files in os.walk("train\\chars2"):
            if len(os.path.basename(root)) > 1:
                continue
            root_int = ord(os.path.basename(root))
            for filename in files:
                filepath = os.path.join(root, filename)
                digit_img = cv2.imread(filepath)
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                chars_train.append(digit_img)
                # chars_label.append(1)
                chars_label.append(root_int)

        chars_train = list(map(deskew, chars_train))
        chars_train = preprocess_hog(chars_train)
        # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
        chars_label = np.array(chars_label)
        print(chars_train.shape)
        self.model.train(chars_train, chars_label)
    if os.path.exists("svmchinese.dat"):
        self.modelchinese.load("svmchinese.dat")
    else:
        chars_train = []
        chars_label = []
        for root, dirs, files in os.walk("train\\charsChinese"):
            if not os.path.basename(root).startswith("zh_"):
                continue
            pinyin = os.path.basename(root)
            index = provinces.index(pinyin) + configMain.PROVINCE_START + 1  # 1是拼音对应的汉字
            for filename in files:
                filepath = os.path.join(root, filename)
                digit_img = cv2.imread(filepath)
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                chars_train.append(digit_img)
                # chars_label.append(1)
                chars_label.append(index)
        chars_train = list(map(deskew, chars_train))
        chars_train = preprocess_hog(chars_train)
        # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
        chars_label = np.array(chars_label)
        print(chars_train.shape)
        self.modelchinese.train(chars_train, chars_label)


def ocrText(charList):
    predict_result = []
    modelchinese = SVM(C=1, gamma=0.5)
    modelchinese.load("svmchinese.dat")
    model = SVM(C=1, gamma=0.5)
    if os.path.exists("svm.dat"):
        model.load("svm.dat")


    for i, part_card in enumerate(charList):
        part_card = preprocess_hog([part_card])
        if i == 0:
            resp = modelchinese.predict(part_card)
            charactor = provinces[int(resp[0]) - configMain.PROVINCE_START]
        else:
            resp = model.predict(part_card)
            charactor = chr(resp[0])
        # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
        part_card_old = part_card
        if charactor == "1" and i == len(charList) - 1:
            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                continue
        predict_result.append(charactor)

    Store.textList=predict_result.copy()
    return predict_result


