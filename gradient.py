"""
    14강. 이미지 Gradient를 이용한 경계 찾기

        이미지 gradient(이미지 경사도 혹은 수학적인 용어로 이미지의 변화율)를 이용한 에지(경계선)를 찾는 방법에 대해 알아 보겠습니다.
        OpenCV는 Sobel, Scharr, Laplacian 이 세가지 타입의 Gradient  필터(High-pass filters;HPF)를 제공합니다.
            Sobel, Scharr 미분 (Sobel and Sharr Derivatives)
            Sobel 오퍼레이터는 가우스 스무딩(Gaussian Smoothing)과 미분연산을 결합한 형태의 연산을 수행함으로써 노이즈에 보다 강력한 저항성을 제공합니다.
            Sobel 오퍼레이션은 세로 방향 또는 가로 방향으로 연산 수행이 가능합니다. cv2.Sobel() 함수는 이미지에 sobel 연산을 수행하는 함수입니다.

            cv2.Sobel(src, ddepth, dx, dy, ksize)
                src: Sobel 미분을 적용할 원본 이미지
                ddepth: 결과 이미지 데이터 타입
                    CV_8U: 이미지 픽셀값을 uint8 로 설정
                    CV_16U: 이미지 픽셀값을 uint16으로 설정
                    CV_32F: 이미지 픽셀값을 float32로 설정
                    CV_64F: 이미지 픽셀값ㅇ르 float64로 설정
                    dx, dy: 각각 x방향, y방향으로 미분 차수 (eg. 1, 0 이면, x 방향으로 1차 미분 수행, y 방향으로 그대로 두라는 의미)
                    ksize: 확장 Sobel 커널의 크기. 1, 3, 5, 7 중 하나의 값으로 설정. -1로 설정되면 3x3 Soble 필터 대신 3x3 Scharr 필터를 적용하게 됨

"""

import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import default_import as selImg
#from default_import import select_img as selImg
#img = cv2.imread(selImg.select_img(6), cv2.IMREAD_GRAYSCALE)


def grad():
    img = cv2.imread(selImg.select_img(6), cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(select_img('2'), cv2.IMREAD_GRAYSCALE)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap = 'gray')
    plt.title('orignal'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap = 'gray')
    plt.title('sobel X'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap = 'gray')
    plt.title('sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()

grad()



