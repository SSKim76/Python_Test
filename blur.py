"""
    12강. 이미지 필터링 - blur

        2D Convolution (Image Filtering)
        1차원 신호와 마찬가지로, 이미지 역시 다양한 low-pass filter(LPF),
        high-pass filter(HPF)를 적용해 필터링 할 수 있습니다.

        LPF 필터는 이미지 노이즈를 제거하거나 이미지를 블러링 하기 위해 사용되며,
        HPF 필터는 이미지에서 edge를 찾는데 활용됩니다.

        OpenCV는 필터 커널을 이미지에 convolve하여 적용하는 cv2.filter2D() 함수를 제공하고 있습니다.
        이미지의 픽셀 값을 해당 픽셀의 이웃과 평균하여 그 값을 취하도록 할 수 있는데,
        이를 averaging filter라고 하며, 이미지의 각 픽셀을 다음과 같은 방법으로 작용하여 픽셀 값을 조정합니다.
            픽셀을 중심으로 5x5 영역을 만듬
            이 영역의 모든 픽셀 값을 더함
            더한 값을 25로 나누고 이 값을 중심 픽셀 값으로 취함
        즉, 이 커널이 적용된 averaging filter는 5x5 영역 내의 모든 픽셀 값들의 평균값을 취합니다.

        >>> blur = cv2.blur(img, (val, val))
            img: 블러링 필터를 적용할 원본 이미지
            (val, val): 필터 커널 사이즈. 두 값이 달라도 무관함

        >>> blur = cv2.GaussianBlur(img, (val, val), 0)
            (val, val): Gaussian 블러 필터. (val1, val2)와 같이 두 개의 값이 달라도 되지만, 모두 양의 홀수이어야 함
            0: sigmaX 값 = 0. sigmaY 값은 자동적으로 0으로 설정되고 Gaussian 블러 필터만을 적용함

        >>> blur = cv2.medianBlur(img, val)
            val: 커널 사이즈. val x val 크기의 박스내에 있는 모든 픽셀들의 median 값을 취해서 중앙에 있는 픽셀에 적용함

            Gaussian Filter는 이미지의 가우스 노이즈를 제거하는데 가장 효과적이며,
            Median Filter는 화면에 소금-후추 노이즈(소금과 후추를 뿌린 듯한 노이즈)를 제거하는데 매우 효과적입니다.

       >>> blur = cv2.bilateralFilter(img, 9, 75, 75)
        cv2.bilateralFilter()
            edge를 보존하고 표면의 질감 등을 제거해주는 매우 효과적인 방법을 제공합니다.
            하지만 이 필터는 다른 필터에 비해 성능적으로 느린 면이 있습니다.
            Guassian 필터와 비교해보면, Gaussian은 하나의 픽셀을 기준으로 이 픽셀 주의에 있는
            픽셀 들의 값들에 의존적으로 계산을 수행하며, 필터링 동안에도 픽셀이 타겟 픽셀과 동일한 값을 가지고 있는 건지,
            픽셀이 모서리에 존재하는지 안하는지 이런 부분은 체크를 하지 않습니다.
            이런 이유로 Gaussian 필터를 적용하여 이미지 처리를 하면, edge가 보존되지 않고 뭉개져 버립니다.

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
        8 : 'img/girl.png'                      # 기본여자 이미지
        9 : 'img/alpha1.jpg'                    # 알파벳 및 숫자 이미지
    }.get(sel_num,'img/girl.png')


def close_window():
    cv2.waitKey(0)
    cv2.destroyAllWindows( )


def onMouse(x):
    pass


def bluring():
    img = cv2.imread(select_img(8))

    #kernel = np.ones((5,5), np.float32)/25
    kernel = np.ones((3, 3), np.float32) / 9
    blur = cv2.filter2D(img, -1, kernel)

    cv2.imshow('original', img)
    cv2.imshow('blur', blur)

    close_window()


def bluring2():
    img = cv2.imread(select_img(8))

    cv2.namedWindow('BlurPanel')
    cv2.createTrackbar('BLUR_MODE', 'BlurPanel', 0, 3, onMouse)
    cv2.createTrackbar('BLUR', 'BlurPanel', 0, 5, onMouse)

    mode = cv2.getTrackbarPos('BLUR_MODE', 'BlurPanel')
    val = cv2.getTrackbarPos('BLUR', 'BlurPanel')

    while True:
        val = val*2 + 1

        try:
            if mode == 0:
                blur = cv2.blur(img,(val, val))
            elif mode == 1:
                blur = cv2.GaussianBlur(img, (val, val), 0)
            elif mode == 2:
                blur = cv2.medianBlur(img, val)
            elif mode == 3:
                blur = cv2.bilateralFilter(img, val*2, 125, 125)
            else:
                break
            cv2.imshow('BlurPanel', blur)
        except:
            break

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        mode = cv2.getTrackbarPos('BLUR_MODE', 'BlurPanel')
        val = cv2.getTrackbarPos('BLUR', 'BlurPanel')

    cv2.destroyAllWindows()

#bluring()
bluring2()





