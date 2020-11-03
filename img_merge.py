"""
    08. 이미지 연산처리를 이용한 이미지 합성하기


>>> img = img1 + img2

위 코드는 img1과 img2를 더해 새로운 이미지를 생성하는 것입니다. 
단 여기서 img1과 img2는 동일한 크기의 동일한 데이터 타입으로 되어 있어야 합니다. 
또는 img2가 하나의 단일한 값을 가져도 됩니다.

이 연산에서 각 픽셀들을 더한 값이 255보다 크면 그 값을 256으로 나눈 나머지가 픽셀값이 됩니다. 
예를 들어 257일 경우, 256으로 나눈 나머지가 1이므로, 픽셀값은 1이 됩니다.

이미지를 더하는 연산에 있어, 위의 예처럼 Numpy array를 그대로 더하는 방법도 있지만, 
OpenCV의 add() 함수를 이용하는 방법도 있습니다.

>>> img = cv2.add(img1, img2)
위 코드도 img1과 img2의 각 픽셀들 값을 더합니다. 
하지만 Numpy array 연산과 다르게 더한 값이 255보다 크면 255로 값이 정해집니다.
"""

import numpy as np
import cv2 as cv2

#import cv2 as cv2
"""

try:
    from numpy import numpy as np
    import cv2.__init__ as cv2
#    from cv2 import cv2
except ImportError:
    pass
"""


def addImage(imgfile1 , imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    cv2.imshow('img1', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows( )

    add_img1 = img1 + img2
    add_img2 = cv2.add(img1, img2)

    cv2.imshow('img1+img2', add_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows( )

    cv2.imshow('add(img1,img2', add_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows( )

addImage("img/sample_img1.png", "img/sample_img2.png")

# Image Blending
#   가중치를 두어 합치는 방법

def onMouse(x):
    pass

def imgBlending(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    cv2.namedWindow('ImgPane')
    cv2.createTrackbar('Mixing', 'ImgPane', 0, 100, onMouse)
    # 트랙바생성  0 ~ 100까지 조절가능

    while True:
        mix = cv2.getTrackbarPos('Mixing', 'ImgPane')
        # mix의 초기값은 0

        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
        cv2.imshow('ImgPane', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

imgBlending("img/sample_img1.png", "img/sample_img2.png")




# 이미지 비트연산
#   이미지 간 비트 연산은 다른 비트 연산과 마찬가지로 AND, OR, NOT, XOR 연산이 있습니다.
#   이미지 비트 연산은 이미지에서 특정 영역을 추출하거나 직사각형 모양이 아닌 ROI를
#   정의하거나 할 때 매우 유용합니다.


def bitOperation(hpos, vpos):
    #hpos, vpos : ROI 의 시작위치
    img1 = cv2.imread("img/sample_img2.png")
    img2 = cv2.imread("img/opencv_logo2.png")

    # 로고를 사진 왼쪽 상단에 두기위해 해당영역 지정하기
    rows, cols, channels = img2.shape
    # 로고 이미지의 크기를 구함

    roi = img1[vpos:rows+vpos, hpos:cols+hpos]
    # ROI 영역(hpos & vpos = 시작좌표, hpos+cols & vpos + rows = 종료좌표표


   # 로고를 위한 마스크와 역마스크 생성하기
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #   흑백으로 변환

    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # 마스크 생성

    mask_inv = cv2.bitwise_not(mask)
    #마스크 반전


    # ROI에서 로고에 해당하는 부분만 검정색으로 만들기
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # cv2.bitwise_and(src1, src2, mask) 는 mask의 값이 0이 아닌 부분만
    # src1과 src2를 AND 연산 합니다.
    # mask의 값이 0인 부분은 mask로 그대로 씌워두는 것이죠.
    #cv2.bitwise_and() 함수의 인자로 사용되는 mask는 1채널 값이어야 하므로
    # 대부분 흑백 이미지입니다.
    # mask의 값이 0이 아닌 부분은 곧 흰색 부분을 말하므로
    # mask의 검정색 부분은 연산을 하지 않고 검정색 그대로 이미지에 놓여지게 됩니다.

    #로고이미지에서 로고 부분만 추출하기
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    #로고 이미지 배경을 cv2.add로 투명으로 만들고 ROI에 로고 이미지 넣기
    dst = cv2.add(img1_bg, img2_fg)
    # 검정색 픽셀값은 0이므로 두 이미지를 더하게 되면 검정색은 없어지고
    # 검정색 아닌 색이 표출되게 되죠~
    # img1_bg에 img2_fg를 합치면 아래와 같은 이미지가 생성됩니다.

    img1[vpos:rows+vpos, hpos:cols+hpos] = dst
    # 이제 로고를 이미지의 ROI 영역에 덮어 쓰면 마무리 됩니다

    cv2.imshow('Result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bitOperation(50, 50)





