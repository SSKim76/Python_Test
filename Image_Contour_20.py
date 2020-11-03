"""
20강. Image Contour 응용3

    이번 강좌에서는 Contour를 활용하여 이미지상의 객체들에 대한 주요 속성들을 도출하는 방법에 대해 살펴봅니다.
    객체들의 주요 속성들에는 아래와 같은 것들이 있습니다.

        Aspect Ratio(종횡비; 가로 세로 비율)
          우리말로 종횡비 또는 가로세로 비율입니다. 주어진 Contour의 외접하는
          직사각형(Bounding Rect)을 구한 후 이 직사각형의 폭과 높이를 이용해서
          Aspect Ration의 값을 구합니다.
          >>> x, y, w, h = cv2.boundingRect(cnt)
          >>> aspect_ratio = float(w)/h

        Extent
           Contour의 넓이와 Contour의 외접 직사각형 넓이의 비로 구할 수 있습니다.
           >>> area = cv2.contourArea(cnt)
            >>> x, y, w, h = cv2.boundingRect(cnt)
            >>> rect_area = w*h
            >>> extent = float(area)/rect_area

        Solidity
            Contour의 넓이와 이 Contour의 Convex Hull 넓이의 비로 구할 수 있습니다.
            >>> area = cv2.contourArea(cnt)
            >>> hull = cv2.convexHull(cnt)
            >>> hull_area = cv2.contourArea(hull)
            >>> solidity = float(area)/hull_area

        Equivalent Diameter
            Contour의 넓이와 동일한 넓이를 가진 원의 지름입니다
            >>> area = cv2.contourArea(cnt)
            >>> equivalent_diameter = np.sqrt(4*area/np.pi)

        Orientation
            Orientation은 객체가 향하고 있는 방향입니다.
            이는 Contour의 최적 타원의 기울어진 각도로 구합니다.
            >>> (x, y), (MajorAxis, MinorAxis), angle = cv2.fitEllipse(cnt)

        Mask and Pixel Points
            가끔 객체를 구성하는 모든 점들, 다시 말하면 모든 픽셀들의 좌표를 추출할 필요가 있을 때가 있습니다.
            이럴 경우 cv2.findNonZero() 함수를 이용하는데, 아래와 같은 방법으로 활용합니다.
            >>> mask = np.zeros(imgray.shape, np.uint8)
            >>> cv2.drawContours(mask, [cnt], 0, 255, -1)
            >>> pixels = cv2.findNonZero(mask)

        Mean Color/Mean Intensity
            Contour 내의 평균 색상, Gray Scale 이미지일 경우 평균 intensity 값을 찾기 위해
            cv2.mean() 함수를 활용하면 됩니다. 위 예제에서 활용된 mask를 그대로 사용합니다.
            >>> mean_value = cv2.mean(img, mask=mask

        Extreme Points
             Contour의 가장 왼쪽점, 가장 오른쪽점, 가장 윗점, 가장 아랫점이라고 보면 되겠습니다.
            Extreme Points는 아래의 코드로 구할 수 있습니다.
            >>> leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            >>> rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            >>> topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            >>> bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
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
    cv2.imshow('thr', thr)
    impDef.close_window()

func(9)


def convex(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))
    img2 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, thr = cv2.threshold(imgray, defThr, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#    cntI = 80
#    for i in contours:
#        cnt0 = contours[cntI]
#        cv2.drawContours(img, [cnt0], 0, (0, 0, 255), 2)
#        cv2.imshow('contour', img)
#        print('cntI = ', cntI)
#        impDef.close_window( )
#        cntI = cntI+1


    cnt = contours[88]

    #cv2.moments() 함수를 이용해 한반도의 무게 중심 좌표를 구합니다
    mmt = cv2.moments(cnt)
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])


    x, y, w, h = cv2.boundingRect(cnt)
    korea_rect_area = w*h
    korea_area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    ellipse = cv2.fitEllipse(cnt)

    aspect_ratio = w/h
    extent = korea_area/korea_rect_area
    solidity = korea_area/hull_area

    print('대한민국 Aspect Ratio : \t%.3f' %aspect_ratio)
    print('대한미국 Extent : \t%.3f' %extent)
    print('대한민국 Solidity : \t%.3f' %solidity)
    print('대한민국 Orientation : \t%.3f' %ellipse[2])
    # Orientation of Korea는 한반도 Contour에 최적 타원의 방향입니다.
    # 이 값은 cv2.fitEllipse() 함수의 리턴값인 ellipse의 3번째 멤버인 ellipse[2] 값이며,
    # 수직방향을 기준으로 타원이 회전한 각도를 나타냅니다.



    equivalent_diameter = np.sqrt(4*korea_area/np.pi)
    korea_radius = int(equivalent_diameter/2)

    # 한반도 무게 중심을 빨간색 원으로 표시를 합니다.
    # 그리고 이 무게중심을 중심으로 한 한반도와 면적이 동일한 빨간색 원을 그렸습니다.
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    cv2.circle(img, (cx, cy), korea_radius, (0, 0, 255), 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.ellipse(img, ellipse, (50, 50, 50), 2)


    # 각 방위의 끝 지점(최 남단, 최 북단,....)
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    print('leftmost = ', leftmost)
    print('rightmost = ', rightmost)
    print('topmost = ', topmost)
    print('bottommost = ', bottommost)

    img = cv2.line(img, leftmost, leftmost, (10,10,10), 10)
    img = cv2.line(img, rightmost, rightmost, (10,10,10), 10)
    img = cv2.line(img, topmost, topmost, (10, 10, 10), 10)
    img = cv2.line(img, bottommost, bottommost, (10, 10, 10), 10)



    cv2.imshow('Korea Features', img)

    impDef.close_window()


convex(21)



