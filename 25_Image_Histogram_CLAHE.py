"""
25강. Image Histogram CLAHE

    Contrast Limited Adaptive Histogram Equalization
        히스토그램 균일화는 생각만큼 좋은 이미지를 얻기 어려움
        원본에 비해 전반적으로 밝아 졌으나 원하는 이미지는 아님. 밝은 부분은 너무 밝아져 알아볼 수 없음

        CLAHE : 이미지를 일정한 크기의 작은 블럭으로 구분하고, 블록별로 히스토그램 균일화를 시행하여 이미지 전체에 대한 균일화를 진행.
        이미지에 노이즈가 있는경우, 노이즈를 감쇠시키는 Cantrast Limiting이라는 기법을 이용.
        타일별로 히스토그햄 균일화가 마무리 되면, 타일간 경계부분은 bilinear interpolation을 적용해 매끈하게 만들어 줌.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


def histogram(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))
    img_grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OpenCV를 이용한 방법
    equ = cv2.equalizeHist(img_grayScale)
    # numpy를 이용하여 구현한 것과 동일한 결과를 리턴 함.
    # 하지만 numpy는 컬러 이미지에도 적용가능하지만, cv2.equalizeHist()는 grayscal 이미지만 가능하면 리턴도 grayscal 이미지 임

    res = np.hstack((img_grayScale, equ))
    #  img_grayScal과 equ를 수평으로 붙임

    cv2.imshow('OpenCV Equalizer', res)
    # cv2.imshow('openCV Equalizer', equ)


    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    # 함수를 이용해 clahe 객체를 생성

    claheImg = clahe.apply(img_grayScale)
    # clahe 객체의 apply() 메소드의 인자로 원본 이미지를 입력하여 CLAHE가 적용된 이미지를 획득

    viewImg = np.hstack((img_grayScale, claheImg))
    # 원본이미지와 수평 붙이기

    cv2.imshow('CLAHE', viewImg)


    impDef.close_window()


histogram(24)
histogram(25)
histogram(10)
histogram(11)