"""
26강. Color Image Histogram

    Color Image = Blue, Green, Red 3차원 좌표
    HSV = Hue(색상), Saturation(채도)
    Hue = 0 ~ 179
    Saturation = 0 ~ 255 까지의 범위

    Color Image에 대한 2D 히스토그램은 가로축으로 Saturation, 세로축으로 Hue 값을 가지는 좌표에 해당하는 픽셀수를 나타낸 것

    cv2.calcHist([hsv], [0,1], None, [180, 256], [0, 180, 0, 256])
        채널값 = [0, 1] : Hue와 Saturation 값을 위해 2개 입력(GrayScale의 경우 0)
        BIN 갯수 = [180, 256] : Hue를 위한 BIN 개수 180, Saturation을 위한 BIN의 개수 256(GrayScale의 경우 256)
        범위 = [0, 180, 0, 256] : Hue는 0 ~ 180 사이, Saturation은 0 ~ 256 사이의 값으로 범위 지정(GrayScale의 경우 [0, 256])
        cv2.calcHist()함수는 180 x 256 크기의 Numpy 배열을 리턴

    np.clip(hist * 0.005 * hscale, 0, 1)
    numpy.clip(a, min, max)
      numpy배열 a의 멤버 값이 min보다 작으면 min, max보다 크면 max로 하는 새로운 numpy배열을 리턴
      예) a = [-2, -3, 1, 3, 4, 5, 0, 1] 일경우, numpy.clip(a, 0, 2)의 결과는 [0, 0, 1, 2, 2, 2, 0, 1]이 리턴됨됨

    hist[:,:,np.newaxis]는 (180, 256)인 배열을 (180, 256, 1)로 변경
    hsvmap의 차원은(180, 256, 3), hist의 차원은(180, 256) 만약 다른 조치를 하지않고, 이 두배열을 곱하면 오류 발생
    그래서 차원이 작은 배열을 큰 차원의 배열로 변경해서 곱하게 됨. np.newaxis 상수가 이런 역할을 함.


"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


def onChange(x):
    global hscale
    hscale = x


def HSVmap( ):
    hsvmap = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsvmap.shape[:2])

    hsvmap[:, :, 0] = h  # 세로축
    hsvmap[:, :, 1] = s  # 가로축
    hsvmap[:, :, 2] = 255  # Value(진하기) 값

    hsvmap = cv2.cvtColor(hsvmap, cv2.COLOR_HSV2BGR)

    #cv2.imshow('HSVmap', hsvmap)
    #impDef.close_window( )

    return hsvmap

def histogram(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0,1], None, [180, 256], [0, 180, 0, 256])

    cv2.imshow('hist2D', hist)
    cv2.waitKey(0)

    plt.imshow(hist, interpolation = 'nearest')
    plt.show()

    impDef.close_window()


def hist2D(ImgNo, defThr = 127):
    img = cv2.imread(impDef.select_img(ImgNo))

    hsvmap = HSVmap()
    hscale = 1

    cv2.namedWindow('hist2D', 0)
    cv2.createTrackbar('scale', 'hist2D', hscale, 32, onChange)

    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])

        hscale = cv2.getTrackbarPos('scale', 'hist2D')

        hist = np.clip(hist*0.005*hscale, 0, 1)
        # hist에 0.005 * hscale값을 곱한 이유는 최종적으로 표시되는 결과에 몇몇 픽셀로 이루어진 값을 제거하기 위해
        hist = hsvmap*hist[:,:,np.newaxis]/255.0
        # hsvmap의 차원은(180, 256, 3), hist의 차원은(180, 256) 만약 다른 조치를 하지않고, 이 두배열을 곱하면 오류 발생
        # 그래서 차원이 작은 배열을 큰 차원의 배열로 변경해서 곱하게 됨. np.newaxis 상수가 이런 역할을 함.
        # hist[:,:,np.newaxis]는 (180, 256)인 배열을 (180, 256, 1)로 변경

        cv2.imshow('hist2D', hist)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


#histogram(26)
#HSVmap()

hist2D(26)




