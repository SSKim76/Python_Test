"""
24강. Image Histogram 균일화

    이미지 히스토그램 이퀠라제이션, 히스토그램 균일화
    좁은 범위에 집중되어있는 픽셀값을 0 ~ 255 범위에 골고루 분포하도록 변환하는 것

    Numpy를 이용하는 방법 : 수학적 지식을 필요
    OpenCV를 이용하는 방법이 있음

"""

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import default_import as impDef


def histogram(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))
    img_grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist, bins = np.histogram(img_grayScale.ravel(), 256, [0,256])
    # Numpy를 이용해 히스토그램 구하기
    
    cdf = hist.cumsum()
    # numpy배열을 1차원 배열로 변환한 후, 각 멤버값을 누적하여 더한 값을 멤버로 하는 1차원 numpy 배열생성

    cdf_m = np.ma.masked_equal(cdf,0)
    # numpy 1차원 배열인 cdf에서 값이 0인 부분은 모두 mask 처리하는 함수. 즉 cdf에서 값이 0인것은 무시
    # numpy 1차원 배열 a가 [1, 0, 0, 2]라면 np.ma.masked_equal(a,0)의 값은 [1, --, --, 2] mask로 처리된 부분은 '--
    ' 으로 표시됨'
    
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # 히스토그램 균일화 방정식을 코드로 표현
    
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # numpy 1차원 배열인 cdf_m에서 마스크된 부분을 0으로 채운 후 numpy 1차원 배열로 리턴
    # 위에서 0을 마스크처리한 부분을 복원
    
    img2 = cdf[img_grayScale]
    # 원래 이미지에 히스토그램을 적용한 새로운 이미지 img2를 생성

    cv2.imshow('Gray Scale', img_grayScale)
    cv2.imshow('Histogram Equalization', img2)
    #cv2.imwrite('img/Histogram_equal.jpg', img2)

    #npImage = np.hstack((img_grayScale, img2))
    #  img_grayScal과 equ를 수평으로 붙임
    #cv2.imshow('numpy Histogram Equalization', npImage)




    # OpenCV를 이용한 방법
    equ = cv2.equalizeHist(img_grayScale)
    # numpy를 이용하여 구현한 것과 동일한 결과를 리턴 함.
    # 하지만 numpy는 컬러 이미지에도 적용가능하지만, cv2.equalizeHist()는 grayscal 이미지만 가능하면 리턴도 grayscal 이미지 임

    res = np.hstack((img_grayScale, equ))
    #  img_grayScal과 equ를 수평으로 붙임

    cv2.imshow('OpenCV Equalizer', res)
    # cv2.imshow('openCV Equalizer', equ)


    impDef.close_window()


histogram(24)
histogram(10)
histogram(11)



