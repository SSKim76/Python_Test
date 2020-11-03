"""
DW Fail Image CLAHE 적용 Test

        CLAHE : 이미지를 일정한 크기의 작은 블럭으로 구분하고, 블록별로 히스토그램 균일화를 시행하여 이미지 전체에 대한 균일화를 진행.
        이미지에 노이즈가 있는경우, 노이즈를 감쇠시키는 Cantrast Limiting이라는 기법을 이용.
        타일별로 히스토그햄 균일화가 마무리 되면, 타일간 경계부분은 bilinear interpolation을 적용해 매끈하게 만들어 줌.

"""

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import os
import importFunc as impFunc


def histogram():

    inPath = "DW_202006/Fail_0630_14/"
    outPath = "DW_202006/Fail_0630_14_Result/"

    fileList = os.listdir(inPath)

    for file in fileList:
        imgName = inPath + file
        img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
        #print(imgName)
        #cv2.imshow('gray', img)

        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
        # 함수를 이용해 clahe 객체를 생성

        claheImg = clahe.apply(img)
        # clahe 객체의 apply() 메소드의 인자로 원본 이미지를 입력하여 CLAHE가 적용된 이미지를 획득


        #cv2.imshow('CLAHE', claheImg)

        outputFilename = outPath+file
        cv2.imwrite(outputFilename, claheImg)
        #print(outputFilename)
        #impDef.close_window()



def conImg():
    inPath = "DW_202006/Fail_0624_1132/"
    outPath = "DW_202006/Fail_0624_1132_Result/"

    fileList = os.listdir(inPath)

    sTime = impFunc.time.time()

    for file in fileList:
        imgName = inPath + file
        img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)

        img2 = impFunc.cvtGamma(img, 3)

        outputFilename = outPath + file
        cv2.imwrite(outputFilename, img2)
    print('Changing Time : ', impFunc.time.time()-sTime)




conImg()
#histogram()



