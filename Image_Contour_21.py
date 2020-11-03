"""
21강. Image Contour 응용4

    Convexity Defects 개요와 이미지에서 Convexity Defects를 찾는 법
    어느 한점에서 도형까지 최단거리 찾기
    서로 다른 모양의 도형 닮음 정도 알아내기

        Convexity Defects : 오목하게 들어가는 모든 부분
        Convex Hull : 볼록하게 나온 부분

        Convexity Defects = cv2.convexityDefects()사용
            ConvexHull을 구하고 cv2.convexityDefects() 함수의 인자로 이미지 contour와 coutour Hull을 비교해서 구함


    cv2.pointPolygonTest()
        이미지상의 어느 한점과 특정 Contour 사이의 거리를 계산하여 리턴
        점이 Contour 외부에 있으면(-), Contour 내부에 있으면 (+) 값을 리턴


    cv2.matchShapes()
        두 도형 또는 두 Contour를 비교하여 닮은 정도를 수치로 리턴(값이 작으면 닮음 정도가 높음)
        만약 두 도형이 완전히 닮은 꼴이면 0이 리턴 됨.
        cv2.matchShape(cnt1, cnt2, method, parameter)
            cnt1, cnt2 : 비교할 두 도형 또는 Contour
            method : 3종류, Hu Moment를 이용해 계산함
                1. cv2.CONTOURS_MATCH_I1 : 현재 버전에서 이 상수가 정의되어 있지 않아 1로 정의해서 사용
                2. cv2.CONTOURS_MATCH_I2 : 현재 버전에서 이 상수가 정의되어 있지 않아 2로 정의해서 사용
                3. cv2.CONTOURS_MATCH_I3 : 현재 버전에서 이 상수가 정의되어 있지 않아 3으로 정의해서 사용
            parameter : 현재 지원하지 않는 인자. 0.0 값으로 입력

        즉 cv2.matchShape(cnt1, cnt2, 1, 0.0)으로 함수 호출

"""

import numpy as np
from cv2 import cv2
import matplotlib as plt
import default_import as impDef


def func(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))
    img1 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray,defThr, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]

    # Convex Hull만을 찾기 위해 contours 1개만을 인자로 입력
    hull = cv2.convexHull(cnt)
    cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)

    # Convexity Defects를 찾기위해 2번째 인자로 returnPoints = False를 지정해야 함
    # (cnt, returnPoints = False) 인자로 주어진 cnt를 분석하여 Convex Hull을 이루는 모든 좌표를 리턴하는 것이 아니라,
    # 원래 Contour와 Convex Hull이 만나는 부부의 Contour 인덱스를 리턴함.
    # 즉 별의 꼭지점에 해당하는 5군데를 리턴함.
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)



    for i in range(defects.shape[0]):
        sp, ep, fp, dist = defects[i, 0]
        start = tuple(cnt[sp][0])
        end = tuple(cnt[ep][0])
        farthest = tuple(cnt[fp][0])

        cv2.circle(img, farthest, 5, (0, 255, 0), -1)

    cv2.imshow('defects', img)
    impDef.close_window()




def PPT(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))
    img2 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, thr = cv2.threshold(imgray, defThr, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)     # contour를 녹색으로 그림

    outside = (55, 70)
    inside = (140, 150)

    # (55, 70)으로 설정된 outside와 (140, 150)으로 설정된 inside에 대해 별보양의 Contour까지 최단거리를 구함
    # cv2.pointPolygonTest()함수의 3번째 인자가 True로 설정되면 Contour와 점 사이의 최단거리를 리턴
    # cv2.pointPolygonTest()함수의 3번째 인자가 False로 설정되면 주어진 점이 Contour의 외부에 있으면 -1, Contour위에 있으면 0,
    # Contour내부에 있으면 1을 리턴
    dist1 = cv2.pointPolygonTest(cnt, outside, True)
    dist2 = cv2.pointPolygonTest(cnt, inside, True)

    print('Contour에서 (%d, %d)까지의 거리 : %.3f' %(outside[0], outside[1], dist1))
    print('Contour에서 (%d, %d)까지의 거리 : %.3f' % (inside[0], inside[1], dist2))

    cv2.circle(img, outside, 3, (0, 255, 255), -1)      # 노란색 점
    cv2.circle(img, inside, 3, (255, 0, 255), -1)       # 분홍색 점

    cv2.imshow('defects', img)
    impDef.close_window()





def MATCH():
    imgfile_list = ['img/star.jpg', 'img/221.jpg', 'img/222.jpg', 'img/223.jpg']

    wins = map(lambda x: 'img'+str(x), range(4))
    wins = list(wins)
    imgs = []
    contour_list = []

    i = 0
    for imgfile in imgfile_list:
        img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
        imgs.append(img)

        ret, thr = cv2.threshold(img, 127, 255, 0)
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list.append(contours[0])
        i+=1

    for i in range(3):
        cv2.imshow(wins[i+1], imgs[i+1])
        ret = cv2.matchShapes(contour_list[0], contour_list[i+1], CONTOURS_MATCH_I1, 0.0)
        print(ret)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


func(22)
PPT(22)

CONTOURS_MATCH_I1 = 1
CONTOURS_MATCH_I2 = 2
CONTOURS_MATCH_I3 = 3

MATCH()



