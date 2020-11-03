"""
23강. Image Histogram 이해하기

    이미지 히스토그램이란 가로축으로 이미지 픽셀값을, 세로축으로 이미지 픽셀수를 좌표에 나타낸 것
    쉽게말해 어떤 이미지에서 밝은 픽셀과 어두운 픽셀의 수가 어느정도 분포하고 있는지 알 수 있는 그래프

    이미지 히스트로그램은 OpenCv 또는 Numpy에서 제공하는 함수를 이용하여 찾을 수 있음음
"""

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import default_import as impDef


def histogram(ImgNo, defThr = 127):
    img1 = cv2.imread(impDef.select_img(ImgNo))
    img2 = cv2.imread(impDef.select_img(ImgNo))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    #OpenCV 함수를 이용해 히스토그램 구하기
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    #   cv2.calcHist(img, channel, mask, histSize, range)
    #       이미지 히스토그램을 찾아서  Numpy배열로 리턴
    #       img : 히스토그램을 찾을 이미지, 인자는 반드시 []로 둘러싸야 함
    #       Channel : grayscal의 경우 [0]을 입력, 컬러이미지의 경우 B,G,R에 대한 [0],[1],[2]를 입력
    #       mask : 이미지 전체의 히스토그램을 구할경우 None, 특정영역을 구할 경우 이 영역에 해당하는 mask값을 입력
    #       histSize : BIN 개수, 인자는 []로 둘러싸야 함.
    #       range : 픽셀값 범위, 보통[0, 256]
    #   히스토그램의 구하기 위해 가장 성능좋은 함수는 cv2.calcHist() 함수.....


    #numpy를 이용해 히스토그램 구하기
    hist2, bins = np.histogram(img1.ravel(), 256, [0, 256])
    #   np.histogram
    #       이미지에서 구한 히스토그램과 BIN의 개수를 리턴

    # 1-D 히스토그램의 경우 Numpy가 빠름
    hist3 = np.bincount(img1.ravel(), minlength = 256)
    #   np.bincount()
    #       Grayscale의 경우 np.historam()보다 약 10배 정도 빠르게 결과를 리턴.
    #       numpy.ravel() : numpy배열을 1차원으로 바꿔주는 함수


    # matplotlib으로 히스토그램 그리기
    plt.hist(img1.ravel(), 256, [0, 256])
    #   plt.hist() 하스토그램을 구하지 않고 바로 그림.
    #   두번째 인자가 BIN의 개수 이 값을 16으로 바꿔주면 BIN의 개수가 16인 히스토그램이 그려짐.

    # 컬러이미지 히스토그램 그리기
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img2], [i], None, [256], [0,256])
        plt.plot(hist, color = col)
        plt.xlim([0,256])
        # 가로축을 0 ~ 256까지로 제한

    plt.show()

    #impDef.close_window()



histogram(23)



