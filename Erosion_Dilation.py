"""
    13강 이미지 Erosion & Dilation
        이미지 Erosion
        이미지 Dilation
        Erosion, Dilation을 이용한 보다 향상된 이미지 변형 처리 기법 (Opening, Closing 등)

        >>> erosion = cv2.erode(img, kernel, iterations=1)
            img: erosion을 수행할 원본 이미지
            kernel: erosion을 위한 커널
            iterations: Erosion 반복 횟수

        >>> dilation = cv2.dilate(img, kernel, iterations=1)

        커널이 크거나 반복회수가 많아지면 erosion일 경우에는 전경의 이미지가 가늘다 못해 없어질 수도 있으며,
        dilation의 경우 그 반대로 될 수 있습니다.

    cv2.morphologyEx() 함수를 이용한 Opening과 Closing
        각종 노이즈를 제거하는데 효과적인데요,
        Opening 기법은 erosion 수행을 한 후 바로 dialation 수행을 하여 본래 이미지 크기로 돌려 놓는 것이고,
        Closing 기법은 dialation 수행을 한 후 바로 erosion 수행을 하여 본래 이미지 크기로 돌려 놓는 것입니다.
        Opening : 검은바탕에 흰점 제거
        Closing : 흰바탕에 검은점 제거
        cv2.MORPH_OPEN: Opening을 수행
        cv2.MORPH_CLOSE: Closing을 수행
        cv2.MORPH_GRADIENT: Dilation 이미지와 Erosion 이미지의 차이를 나타냄
        cv2.MORPH_TOPHAT: 원본 이미지와 opening한 이미지의 차이를 나타냄
        cv2.MORPH_BLACKHAT: closing한 이미지와 원본 이미지의 차이를 나타냄

    cv2.getStructuringElement() 함수를 이용한 커널 매트릭스 만들기
        앞에서 설명했던 예제에서 Numpy의 도움을 받아 우리가 커널 매트릭스를 생성했습니다.
        하지만 경우에 따라서 원, 타원 모양의 커널을 만들어 적용해야 할 필요가 있는데요,
        이러한 매트릭스를 우리가 스스로 생성해서 사용해도 되지만
        cv2.getStructuringElement() 함수를 이용하면 손쉽게 만들 수 있습니다.
        이 함수에 여러분이 원하는 커널 모양과 커널 크기만 인자로 넘겨주면 이 함수가 커널 매트릭스를 알아서 만들어줍니다.

        >>> M1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            직사각형 모양으로 5x5 크기의 커널 매트릭스를 생성합니다.
            cv2.MORPH_ELLIPSE: 타원 모양으로 매트릭스를 생성
            cv2.MORPH_CROSS: 십자 모양으로 매트릭스를 생성

"""

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt


def select_img(sel_num):
    return{
        0 : 'img/100.jpg',                 # 경현
        1 : 'img/images_1.jpg',          # 그라데이션 이미지
        2 : 'img/multi_light.jpg',          #
        3 : 'img/opencv_logo.jpg',
        4 : 'img/sample_img1.jpg',       # 산과 강이 있는 풍경
        5 : 'img/sample_img2.jpg',       # 수지 이미지
        6 : 'img/sdoku.jpg',                  #
        7 : 'img/tile.jpg',                     # Threshold Test용 타일
        8 : 'img/girl.png',                      # 기본여자 이미지
        9 : 'img/alpha1.jpg',                    # 알파벳 및 숫자 이미지
        10: 'img/noise_alp.jpg'
    }.get(sel_num,'img/girl.png')


def close_window():
    cv2.waitKey(0)
    cv2.destroyAllWindows( )


def morph():
    img = cv2.imread(select_img(9))

    kernel = np.ones((3,3), np.uint8)
    # 3x3 크기의 1로 채워진 매트릭스를 생성. Erosion 및 Dilation 커널로 사용예정.

    erosion = cv2.erode(img, kernel, iterations = 1)
    dilation = cv2.dilate(img, kernel, iterations = 1)

    cv2.imshow('original', img)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)

    close_window()


def morph1():
    img = cv2.imread(select_img(10), cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((4,4), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('original', img)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    close_window()


#morph()
morph1()








