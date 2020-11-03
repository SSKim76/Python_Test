"""
DW Fail Image CLAHE 적용 Test

        CLAHE : 이미지를 일정한 크기의 작은 블럭으로 구분하고, 블록별로 히스토그램 균일화를 시행하여 이미지 전체에 대한 균일화를 진행.
        이미지에 노이즈가 있는경우, 노이즈를 감쇠시키는 Cantrast Limiting이라는 기법을 이용.
        타일별로 히스토그햄 균일화가 마무리 되면, 타일간 경계부분은 bilinear interpolation을 적용해 매끈하게 만들어 줌.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import importFunc as impFunc


def onMouse(x):
    pass

def onChange(x):
    global clipLimit
    clipLimit = x

def onChange1(x):
    global tileGrideSize
    tileGrideSize = x

def imgTest(fileName):

    minClipLimit = 1
    maxClipLimit = 16
    minGridSize = 2
    maxGridSize = 32

    clipLimit = 1
    tileGrideSize = 2


    imgName = "C:/Users/JST-0602/PycharmProjects/Python_OpenCV/OpenCV/DW_Test/DW_Img/" + fileName
    #print(imgName)

    cv2.namedWindow('CLAHE_Test')
    cv2.createTrackbar('clipLimit', 'CLAHE_Test', minClipLimit, maxClipLimit, onChange)
    cv2.createTrackbar('tileGrideSize', 'CLAHE_Test', minGridSize, maxGridSize, onChange1)

    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('CLAHE_Test', img)
    #cv2.waitKey(0)

    while True:

        clipLimit = cv2.getTrackbarPos('clipLimit', 'CLAHE_Test')
        tileGrideSize = cv2.getTrackbarPos('tileGrideSize', 'CLAHE_Test')

        print('clipLimit = ', clipLimit)
        print('tileGrideSize = ', tileGrideSize)

        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = (tileGrideSize,tileGrideSize))
        # 함수를 이용해 clahe 객체를 생성

        claheImg = clahe.apply(img)
        # clahe 객체의 apply() 메소드의 인자로 원본 이미지를 입력하여 CLAHE가 적용된 이미지를 획득

        cv2.imshow('CLAHE_Test', claheImg)


        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


#imgTest("K-001.jpg")
#imgTest("K-003.jpg")
#imgTest("K-007.jpg")







