import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import stark
import Store
import threading
import time
# import main  不能重复导入  A导入B  B导入A

def showUI(begin):
    win = tk.Tk()
    surface = Surface(win,begin)
    win.protocol('WM_DELETE_WINDOW', lambda: close_window(surface, win))
    win.mainloop()


class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600

    def __init__(self, win,begin):
        self.begin=begin
        ttk.Frame.__init__(self, win)
        win.geometry('200x100')
        frame_right2 = ttk.Frame(self)
        win.title("车牌识别")
        win.state("zoomed")
        self.pack()

        frame_right2.pack()

        # ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)
        var = tk.StringVar()  # 这时文字变量储存器

        from_pic_ctl = ttk.Button(frame_right2, text="选择图片", width=20, command=self.from_pic)
        from_vedio_ctl = ttk.Button(frame_right2, text="显示过程", width=20, command=self.from_vedio)
        button = ttk.Button(frame_right2, text="显示结果", width=20, command=lambda:self.showResult(var))


        l = tk.Label(win,
                     textvariable=var,  # 使用 textvariable 替换 text, 因为这个可以变化
                     bg='green', font=('Arial', 12), width=15, height=2)
        l.pack()

        from_vedio_ctl.pack(anchor="se", pady="5")
        from_pic_ctl.pack(anchor="se", pady="5")
        button.pack(anchor="se", pady="5")

    def showResult(self,var):
        Str = ""
        for s in Store.textList:
            Str = Str + s

        for c in Store.colors:
            Str = Str + "\n" + c + "车牌"
            break

        var.set(Str)


    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk


    def from_vedio(self):
        stark.showProcess()

    def from_pic(self):
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
        if self.pic_path:
            self.begin(self.pic_path)
            # cv2.namedWindow('originImg',cv2.WINDOW_NORMAL)
            # cv2.imshow('originImg',Store.resizedImg)
            # cv2.waitKey(0)
            # self.show_roi(r, roi, color)


def close_window(surface, win):
    print("destroy")
    win.destroy()
