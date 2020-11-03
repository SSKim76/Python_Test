"""
31강. Template Matching2

    템플릿 매칭을 이용하여 하나의 이미지에서 동일한 객체를 모두 찾는 방법

        cv2.minMaxLoc() 함수는 이미지에 나타나는 모든 동일한 객체에 대한 위치를 제공하지 않음

"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef

# ImgNo 원본이미지에서 모든 tempNo(template Image)를 찾는다.
# thr값을 높이면 이미지의 정확도가 올라간다.
def tmpMatch(ImgNo, tempNo, thr):
    img = cv2.imread(impDef.select_img(ImgNo))
    #imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.imread(impDef.select_img(ImgNo), cv2.IMREAD_GRAYSCALE)

    template = cv2.imread(impDef.select_img(tempNo), cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(imgray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= thr)
"""
    loc = np.where(res>=threshold)
    cv2.matchTemplate() 함수의 리턴값 res에서 threshold 보다 큰 값을 가진 위치들을 튜플로 리턴.
    Numpy.where() 함수를 이해하기 위해 아래의 코드를 한번 볼까요?

    x = np.arange(100)  # 0~99를 멤버로 하는 1차원 배열을 만듬
    x = x*10
    x
        [  0  10  20  30  40  50  60  70  80  90 100 110 120 130 140 150 160 170
         180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350
         360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510 520 530
         540 550 560 570 580 590 600 610 620 630 640 650 660 670 680 690 700 710
         720 730 740 750 760 770 780 790 800 810 820 830 840 850 860 870 880 890
         900 910 920 930 940 950 960 970 980 990]

    loc = np.where(x > 550)
    loc
        (array([56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),)
                
    그런데 코드에서 res는 2차원 numpy 배열임. 
    따라서 리턴되는 위치값도 찾은 row들, 찾은 col들로 리턴 됨. 예제에서 loc는 다음 같다.
    loc = (array([ 73,  74, 128, 128, 128, 183, 183, 183, 237, 237, 251, 278, 279, 292]),
           array([387, 387,  41, 280, 281,  54,  55, 174, 147, 148, 427, 427, 427,  41]))
    
    array([73, 74,,,]) 는 res>=threshold를 만족하는 row의 위치,
    array([387, 387, ...])는 res>=threshold를 만족하는 column의 위치.
    row는 y좌표의 값이고, column은 x좌표의 값임을 유의 
"""

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 2)

"""
    for pt in zip(*loc[::-1])
    파이썬 내장함수 zip은 동일한 개수를 가진 리스트나 튜플을 같은 위치의 멤버들끼리 묶어서 튜플로 만든 다음
    이를 멤버로 하는 리스트로 만들어 줌.
     zip([0, 1, 2, 3], [4, 5, 6, 7])
     [(0, 4), (1, 5), (2, 6), (3, 7)]
    따라서 zip(*loc[::-1])은 loc의 순서를 거꾸로 하여 튜플로 묶고 리스트로 만들어 줍니다.
"""

    cv2.imshow('res', img)

    impDef.close_window()


tmpMatch(30, 31, 0.95)
