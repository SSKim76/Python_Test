"""
    15강. Canny Edge Detection
        가장 인기있는 에지찾기 알고리즘 중 하나
        1단계 : 노이즈제거(가우시안필터 사용)
        2단계 : Gradient 값이 높은 부분 찾기
        3단계 : 최대값이 아닌 픽셀의 값을 0으로 만들기
        4단계 : Hyteresis Thresholding(실제 에지인지 아닌지 판단하는 단계)

        cv2.Canny(img, minThreshold, maxThreshold) -> Canny Edge Detection 알고리즘 구현 함수
            img : 원본이미지 gray scale
            minThreshold : minimum Thresholding value
            maxThreshold : maximum Thresholding value
"""

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import default_import as selImg

#from default_import import select_img as selImg
#img = cv2.imread(selImg.select_img(6), cv2.IMREAD_GRAYSCALE)

def canny():
    img = cv2.imread(selImg.select_img(8), cv2.IMREAD_GRAYSCALE)

    edge1 = cv2.Canny(img, 50, 200)
    edge2 = cv2.Canny(img, 100, 200)
    edge3 = cv2. Canny(img, 170, 200)

    cv2.imshow('original', img)
    cv2.imshow('Canny Edge1', edge1)
    cv2.imshow('Canny Edge2', edge2)
    cv2.imshow('Canny Edge3', edge3)

    selImg.close_window()

canny()

