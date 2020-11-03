"""
19. Image Contour 응용2


"""


import numpy as np
from cv2 import cv2
import matplotlib as plt
import default_import as impDef


def convex(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))
    img1 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray,defThr, 255, 0)
    cv2.imshow('thr', thr)
    impDef.close_window()

    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]

## 코드영역 따기
    cntI = 0;
    for i  in contours:
       cnt0 = contours[cntI]
       area = cv2.contourArea(cnt0)
       print('면적 : ', area)

       #영역의 크기가 45000 보다 크고 50000보다 작은경우만 출력
       if area >= 45000 and area < 50000:
           cv2.drawContours(img1, [cnt0], 0, (0, 0, 255), 2)
           cv2.imshow('contour', img1)
           impDef.close_window()
           
       cntI = cntI +1
##

    cv2.drawContours(img, [cnt], 0, (0,255,0),3)

    check = cv2.isContourConvex(cnt)
    # cv2.isContourConvex() 함수는 인자로 입력된 Contour가 Convex Hull 인지 체크합니다.
    # 만만 Convex Hull이라면 True를 리턴하고 그렇지 않으면 False를 리턴합니다.
    if not check:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img1, [hull], 0, (0,255,0), 3)
        cv2.imshow('convexhull', img1)
        # check 값이 False인 경우, 다시 말하면 우리가 주목하는 Contour가 Convex Hull이
        # 아니라면 cv2.convexHull() 함수를 이용해 원본이미지의 contours[1]에 대한
        # convex hull 곡선을 구합니다.

    cv2.imshow('contour', img)

    impDef.close_window()

def convex1(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray, defThr, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 원본 이미지의 6번째 contour인 contours[5]가 단풍잎 모양의 외곽을 에워싸고 있는 contour
    cnt = contours[5]

    # contour에 외접하는 똑바로 세워진 사각형을 얻기 위해 cv2.boundingRect() 함수를 이용합니다.
    x, y, w, h = cv2.boundingRect(cnt)
    #cv2.boudingRect()함수는 인자로 받은 contour에 외접하고 똑바로 세워진 직사각형의
    # 좌상단 꼭지점 좌표 (x, y)와 가로 세로 폭을 리턴합니다.
    # 이렇게 얻은 좌표를 이용해 원본 이미지에 빨간색으로 사각형을 그립니다.
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 3)
   
    rect = cv2.minAreaRect(cnt)
    #cv2.minAreaRect() 함수는 인자로 입력한 contour에 외접하면서 면적이 가장 작은 직사각형을 구하는데 활용됩니다.
    # 이 함수의 리턴값은 좌상단 꼭지점 좌표 (x, y), 가로 세로 폭과 이 사각형이 기울어진 각도입니다.
    box = cv2.boxPoints(rect)
    #v2.boxPoints() 함수는 cv2.minAreaRect() 함수로 얻은 직사각형의 꼭지점 4개의 좌표를 얻기 위해 사용됩니다.
    box = np.int0(box)
    #좌표는 float형으로 리턴되므로 np.int0()로 정수형 값으로 전환한 후, 원본 이미지에 초록색 사각형을 그리는 겁니다.

    cv2.drawContours(img, [box], 0, (0, 255, 2), 3)
    cv2.imshow('retangle', img)

    impDef.close_window()

convex(10, 50)
convex1(19)



