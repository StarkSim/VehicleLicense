import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys
import os
import Store
import time
import config
import tkinter as tk


import Store

configMain = config.config


imgStore=Store.Store()

def returnGrayImg(imgPath):
    RGBImg = cv2.imread(imgPath, cv2.IMREAD_COLOR)

    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    Store.originImg = img.copy()

    higth, width = img.shape[:2]
    rate = configMain.maxWidth / width
    Store.picWid = img.shape[0]
    Store.picHei = img.shape[1]
    if width > configMain.maxWidth:
        img = cv2.resize(img, (configMain.maxWidth, int(higth * rate)), interpolation=cv2.INTER_AREA)
        RGBImg = cv2.resize(RGBImg, (configMain.maxWidth, int(higth * rate)), interpolation=cv2.INTER_AREA)

    Store.resizedImg = img.copy()
    Store.RGBImg = RGBImg.copy()

    return img

def returnDetail(img):
    kernel = np.ones((20, 20), np.uint8)
    imgLoseDetail = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    imgDetail = cv2.addWeighted(img, 1, imgLoseDetail, -1, 0)

    Store.imgLoseDetail = imgLoseDetail.copy()
    Store.imgDetail = imgDetail.copy()

    return imgDetail

def dropEdgePointByWH_ratio(edgePointList):
    newEdgePointList = []

    for cnt in edgePointList:
        rect = cv2.minAreaRect(cnt)
        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        # print(wh_ratio)
        # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if wh_ratio > 2 and wh_ratio < 5.5:
            newEdgePointList.append(cnt)

    return newEdgePointList


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

def showResult(var):
    Str=""
    for s in Store.textList:
        Str=Str+s



    for c in  Store.colors:
        Str=Str+"\n"+c+"车牌"
        break

    var.set(Str)


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


def showProcess():

    cv2.namedWindow('1',cv2.WINDOW_NORMAL)

    cv2.imshow("1", Store.originImg )
    cv2.waitKey(0)

    cv2.imshow("1",Store.resizedImg)
    cv2.waitKey(0)

    cv2.imshow("1", Store.blurImg )
    cv2.waitKey(0)

    cv2.imshow("1", Store.imgLoseDetail)
    cv2.waitKey(0)

    cv2.imshow("1", Store.imgDetail)
    cv2.waitKey(0)

    cv2.imshow("1",Store.imgEdge)
    cv2.waitKey(0)

    temp=cv2.cvtColor(Store.resizedImg,cv2.COLOR_GRAY2BGR).copy()
    for box in Store.boxPointList:
        cv2.drawContours(temp, [box], 0, (0, 0, 255), 2)
        cv2.namedWindow('1',cv2.WINDOW_NORMAL)
        cv2.imshow('1',temp)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    for card in Store.card_imgs:
        try:
            cv2.imshow('1', card)
        except cv2.error:
            continue
        cv2.waitKey(0)

    for char in Store.charList:
        try:
            cv2.imshow('1', char)
        except cv2.error:
            continue
        cv2.waitKey(0)


from tkinter import StringVar, IntVar
def UIshow():


    window = tk.Tk()
    window.title('数字图像大作业')
    window.geometry()
    img=Store.RGBImg.copy()

    higth, width = img.shape[:2]
    rate = 500 / width

    if width > 500:
        img = cv2.resize(img, (500, int(higth * rate)), interpolation=cv2.INTER_AREA)

    cv2.imwrite('temp.png',img)

    higth, width = img.shape[:2]
    canvas = tk.Canvas(window, bg='green', height=higth, width=width)

    image_file = tk.PhotoImage(file='temp.png')
    image = canvas.create_image(250, 0, anchor='n', image=image_file)  # 图片锚定点（n图片顶端的中间点位置）放在画布（250,0）坐标处



    var = tk.StringVar()  # 这时文字变量储存器
    l = tk.Label(window,
                 textvariable=var,  # 使用 textvariable 替换 text, 因为这个可以变化
                 bg='green', font=('Arial', 12), width=15, height=2)
    l.pack()
    b = tk.Button(window, text='显示结果', command=lambda: showResult(var)).pack()
    b2= tk.Button(window, text='显示过程', command=showProcess).pack()
    canvas.pack()
    # 第7步，主窗口循环显示
    window.mainloop()
