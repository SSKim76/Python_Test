"""
32강. 허프변환을 이용한 직선찾기

    OpenCV를 활용한 허프변환
        CV2.HoughLines() : 배열을 리턴(거리는 픽셀단위, 각도는 라디안 단위)
            첫번째 인자 : 바이너리 이미지(threshold를 적용하거나, Cany Edge Detection 결과를 입력)
            두번째와 세번째 인자 : 거리와 각도의 정확도를 입력
            네번째 인자 : 직선이라고 판단할 수 있는 카운팅 개수를 입력(이 값에 따라 정확도가 달라짐)

    확률적 허프변환
        허프변환은 모든 픽셀에 대해 계산을 하므로 비효율적일 수 있다.
        확률적 허프변환은 무작위로 픽셀을 선택하여 이에 대해 허프변환을 수행하는 것만으로 충분히 직선을 찾을 수 있다.

        cv2.HoughLinesP(edges, 1, np.pi/180, thr, minLineLength, maxLineGap)
            확률적 허프변환을 수행하여 찾고자 하는 직선을 리턴
            minLineLength : 이 값이하로 주어진 선 길이는 자고자 하는 직선으로 간주하지 않는다.
            maxLingGap : 찾은 직선이 이 값 이상으로 떨어져 있으면 다른 직선으로 간주한다.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


#허프변환
def hough(ImgNo, thr):
    img = cv2.imread(impDef.select_img(ImgNo))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray, 50, 150, apertureSize = 3)
    # Canny Edge Detection결과를 구한다. (GrayScale Image, Threshold Min, Threshold Max)

    lines = cv2.HoughLines(edges, 1, np.pi/180, thr)
    # 입력된 이미지의 모든 픽셀에서 Hough변환을 계산, 거리는 1Pixel로 나타내고, 각 픽셀에서도 2도 단위로 직선의 r과 theta를 구한다.
    # 2도 단위이므로 각 픽셀별로 180개의 r과 theta가 나온다.
    # threshold갑으로 주어진 값 이상으로 카운팅된 r과 theta에 대해 lines로 리턴

    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('res', img)
    impDef.close_window()

#확률적 허프변환
def hought_1(ImgNo, thr, minLineLength, maxLineGap):
    img = cv2.imread(impDef.select_img(ImgNo))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray, 50, 150, apertureSize = 3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, thr, minLineLength, maxLineGap)
    # 확률적 허프변환을 수행하여 찾고자 하는 직선을 리턴
    # minLineLength : 이 값이하로 주어진 선 길이는 자고자 하는 직선으로 간주하지 않는다.
    # maxLingGap : 찾은 직선이 이 값 이상으로 떨어져 있으면 다른 직선으로 간주한다.

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('res', img)
    impDef.close_window( )


hough(10, 200)
hough(11, 100)
#hought_1(32, 140, 150, 10)


