

#  代码借鉴自 https://github.com/wzh191920/License-Plate-Recognition

import binger
import stark
import config

import tkinter as tk
import UIPart


def begin(imgPath):
    config.configInit()

    img = stark.returnGrayImg(imgPath)
    img = binger.returnGaussianBlurImg(img)
    img = stark.returnDetail(img)
    edgeImg = binger.returnEdegImg(img)
    edgePointList = binger.returnEdgePointList(edgeImg)
    # print(len(edgePointList))

    edgePointList = binger.dropEdgePointByMinArea(edgePointList)

    edgePointList = stark.dropEdgePointByWH_ratio(edgePointList)

    rectangleList, boxPointList = binger.trunEdgePointToRectAndBoxPoint(edgePointList)

    card_imgs = stark.returnCardImgList(rectangleList)

    card_imgs, colorList = binger.dropCardImgByColorAndReturnColorList(card_imgs)

    card_imgs, colorList = stark.accurateCardPlaceByColor(card_imgs, colorList)

    charList = binger.separateChar(card_imgs, colorList)

    textList = binger.ocrText(charList)

    # nothing = stark.UIshow()

    for text in textList:
        print(text)



UIPart.showUI(begin)




# import cv2
# import numpy as np
# for box in boxPointList:
#     cv2.drawContours(temp, [box], 0, (0, 0, 255), 2)
#
#
#
#     cv2.namedWindow('1',cv2.WINDOW_NORMAL)
#     cv2.imshow('1',temp)
#     cv2.waitKey(0)


# pytesseract.pytesseract.tesseract_cmd = r'D:\Myapp_SubFolder\tesseract-ocr\tesseract.exe'
# ocr = CnOcr()
#
#
# for card in charList:
#     rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
#     results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
#     print("Text: {}".format(results["text"][0]))
#
#     res=ocr.ocr(card)
#     print(res)


# for card in charList:
#     try:
#         cv2.imshow('1', card)
#     except cv2.error:
#         continue
#     cv2.waitKey(0)

# print(len(rectangleList))
#
#
# temp=cv2.cvtColor(Store.resizedImg,cv2.COLOR_GRAY2BGR)
# for box in boxPointList:
#     cv2.drawContours(temp, [box], 0, (0, 0, 255), 2)
#
#
#
#     cv2.namedWindow('1',cv2.WINDOW_NORMAL)
#     cv2.imshow('1',temp)
#     cv2.waitKey(0)
#
# cv2.namedWindow('1',cv2.WINDOW_NORMAL)
# cv2.imshow('1',Store.originImg)
# cv2.waitKey(0)

