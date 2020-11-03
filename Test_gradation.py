"""
    Test. 조명의 영향으로 그라데이션 효과가 생긴 이미지에서 그라데이션 제거하기

        1. Color Image Open
        2. RGB To LAB : Image 변환(Color -> LAB), cv2.cvtColor(cv2.RGB2LAB) or cv2.BGR2LAB
        3. Median Filter : Radius 값을 통해 조명의 상태를 얻어냄 radius 100(보통 20이상 50이하에서도 충분함)
        4. Invert Lightness : 3에서 알아낸 조명정보를 반전시켜 역조명 채널을 만든다.
        5. Composition : 원본영상과 합성하여 이미지를 생성한다.
        6. Histogram 최대-최소 평균으로 Global Thresholding

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef
import os


global img, LabImg, GrayImg, filterImg

#inPath = "DW/5. Full_Color_G_White_Wide_Back/"
inPath = 'gradation_img/1/'
#outPath = "DW/4. Full_Color_G_Result/"
outPath = inPath

sampleImg = ['gradation_img\E06_03_17_265.jpg', 'gradation_img\E06_04_24_406.jpg','gradation_img\E06_04_33_781.jpg',
         'gradation_img\E06_05_39_437.jpg', 'gradation_img\E06_11_40_546.jpg', 'gradation_img\E06_11_53_781.jpg',
         'gradation_img\E06_12_13_265.jpg']


def openImage(imgNo):
    global img, GrayImg, LabImg
    #img = cv2.imread(impDef.select_img(imgNo), cv2.IMREAD_GRAYSCALE)

    if type(imgNo) == int:
        img = cv2.imread(impDef.select_img(imgNo))
    else :
        img = cv2.imread(imgNo)

    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def threshold(threshold, value):
    global img, GrayImg, LabImg
    ret, thr = cv2.threshold(img, threshold, value, cv2.THRESH_BINARY)
    cv2.imshow('origina - Threshold', thr)
    impDef.close_window()


def adaptiveGaussian(value = 55):
    global img, GrayImg, LabImg
    thr3 = cv2.adaptiveThreshold(GrayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value, 2)
    cv2.imshow('Adaptive Gaussian', thr3)
    impDef.close_window()
    cv2.imwrite('img\Adaptive_Gaussian.jpg', thr3)


def adaptiveMean(value = 55):
    global img, GrayImg, LabImg
    thr3 = cv2.adaptiveThreshold(GrayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, value, 2)
    cv2.imshow('Adaptive Mean', thr3)
    impDef.close_window()
    cv2.imwrite('img\Adaptive_Mean.jpg', thr3)


def Otsu():
    global img, GrayImg, LabImg
    ret, thr2 = cv2.threshold(GrayImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('OTSU', thr2)
    impDef.close_window()
    cv2.imwrite('img\Adaptive_OTSU.jpg', thr2)


def GaussianOtsu():
    global img, GrayImg, LabImg
    blur = cv2.GaussianBlur(img, (5,5),0)
    ret, thr1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('Gaussian OTSU', thr1)
    impDef.close_window()
    cv2.imwrite('img\Adaptive_GaussianOTSU.jpg', thr1)


#영상 Open - 100 : Vericode
#openImage(100)

# Threshold 기준값 60 이상이면 White로 변경
#threshold(60, 255)

#adaptiveGaussian(55)
#adaptiveMean(55)
#Otsu()
#GaussianOtsu()


def lightEffect(img_BGR, FileName):
    global img, GrayImg, LabImg, outPath

    # RGB To LAB : Image 변환(Color -> LAB), cv2.cvtColor(cv2.RGB2LAB)
    LabImg = cv2.cvtColor(img_BGR, cv2.COLOR_RGB2LAB)
    GrayImg = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    print('%s RGB To Lab'%FileName)
     #cv2.imshow('LabImg_RGB2LAB', LabImg)
    #impDef.close_window( )

    #Median Filter
    filterImg = cv2.medianBlur(LabImg, 5)
    print('%s medianBlur 5'%FileName)
    #cv2.imshow('MedianFilter Img', filterImg)
    #impDef.close_window( )

    # Reverse Image
    filterImg = cv2.cvtColor(filterImg, cv2.COLOR_RGB2GRAY)
    revImg = cv2.bitwise_not(filterImg)
    print('%s Reverse'%FileName)

    #cv2.imshow('Reverse Image', revImg)
    #impDef.close_window()
    #cv2.imwrite('img/revImg.jpg', revImg)

    # Composite

    # Add
    finalImg = cv2.add(GrayImg, filterImg)
    #finalImg = cv2.add(GrayImg, cv2.cvtColor(revImg, cv2.COLOR_RGB2GRAY))
    cv2.imshow('Final Add Image', finalImg)
    impDef.close_window()
    fileName = outPath+FileName+'_final.jpg'
    print('%s 저장'%FileName)
    cv2.imwrite(fileName, finalImg)

    #Final Threshold
    ret, finThr = cv2.threshold(finalImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('final Threshold Otsu', finThr)
    impDef.close_window()
    fileName = outPath+FileName+'finalThr_Otus.jpg'
    print('%s 저장'%FileName)
    cv2.imwrite(fileName, finThr)

"""
for i in sampleImg:

    print(i)
    viewimg = cv2.imread(i)
    cv2.imshow('원본 Image', viewimg)
    GrayImg = cv2.cvtColor(viewimg, cv2.COLOR_BGR2GRAY)
    impDef.close_window()
    # threshold(60, 255)

    # adaptiveGaussian(55)
    thr3 = cv2.adaptiveThreshold(GrayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 2)
    cv2.imshow('Adaptive Gaussian', thr3)
    impDef.close_window( )
    saveFileName = i+'_Gaussian.jpg'
    cv2.imwrite(saveFileName, thr3)

    # adaptiveMean(55)
    thr2 = cv2.adaptiveThreshold(GrayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 2)
    cv2.imshow('Adaptive Mean', thr2)
    impDef.close_window()
    saveFileName = i+'_Mean.jpg'
    cv2.imwrite(saveFileName, thr2)
    

"""

fileList = os.listdir(inPath)

for file in fileList:
    imgName = inPath + file
    print(imgName)
    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    img_BGR = cv2.imread(imgName, cv2.IMREAD_COLOR)
    cv2.imshow(imgName, img)
    impDef.close_window()

    # adaptiveGaussian(55)
    thr3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 195, 2)
    cv2.imshow('Adaptive Gaussian', thr3)
    impDef.close_window( )

    # Size 변환
    thr3 = cv2.resize(thr3, dsize = (640, 480), interpolation =cv2.INTER_AREA)
    saveFileName = outPath + file + '_Gaussian.jpg'
    print(saveFileName)
    cv2.imwrite(saveFileName, thr3)


    # adaptiveMean(55)
    thr2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 195, 2)
    cv2.imshow('Adaptive Mean', thr2)
    impDef.close_window( )

    # Size 변환
    thr2 = cv2.resize(thr2, dsize = (640, 480), interpolation = cv2.INTER_AREA)
    saveFilename = outPath+file+'_mean.jpg'
    print(saveFileName)
    cv2.imwrite(saveFilename, thr2)

    #Light Effect
    print(file)
    lightEffect(img_BGR, file)
