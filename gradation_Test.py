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
#from default_import import select_img as selImg
#img = cv2.imread(selImg.select_img(6), cv2.IMREAD_GRAYSCALE)


def grad(img):
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



def thresholding(img, threshold = 127, value = 255):

    #img = cv2.imread(impDef.select_img(ImgNo), cv2.IMREAD_GRAYSCALE)

    ret, thr9 = cv2.threshold(img, threshold, value, cv2.THRESH_BINARY)
    thr10 = cv2.adaptiveThreshold(img, value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thr11 = cv2.adaptiveThreshold(img, value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    title = ['original', 'Global Thresholding', 'Adaptive MEAN', 'Adaptive GAUSSIAN']
    images = [img, thr9, thr10, thr11]
    grad(img)
    grad(thr9)
    grad(thr10)
    grad(thr11)

    for i in range(4):
        cv2.imshow(title[i], images[i])



    cv2.waitKey(0)
    cv2.destroyAllWindows()




def histogram(img, defThr = 127):
    #img = cv2.imread(impDef.select_img(ImgNo))
    #img_grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grayScale = img

    hist, bins = np.histogram(img_grayScale.ravel(), 256, [0,256])
    # Numpy를 이용해 히스토그램 구하기
    
    cdf = hist.cumsum()
    # numpy배열을 1차원 배열로 변환한 후, 각 멤버값을 누적하여 더한 값을 멤버로 하는 1차원 numpy 배열생성

    cdf_m = np.ma.masked_equal(cdf,0)
    # numpy 1차원 배열인 cdf에서 값이 0인 부분은 모두 mask 처리하는 함수. 즉 cdf에서 값이 0인것은 무시
    # numpy 1차원 배열 a가 [1, 0, 0, 2]라면 np.ma.masked_equal(a,0)의 값은 [1, --, --, 2] mask로 처리된 부분은 '--
    ' 으로 표시됨'
    
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # 히스토그램 균일화 방정식을 코드로 표현
    
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # numpy 1차원 배열인 cdf_m에서 마스크된 부분을 0으로 채운 후 numpy 1차원 배열로 리턴
    # 위에서 0을 마스크처리한 부분을 복원
    
    img2 = cdf[img_grayScale]
    # 원래 이미지에 히스토그램을 적용한 새로운 이미지 img2를 생성

    cv2.imshow('Gray Scale', img_grayScale)
    cv2.imshow('Histogram Equalization', img2)
    #cv2.imwrite('img/Histogram_equal.jpg', img2)

    #npImage = np.hstack((img_grayScale, img2))
    #  img_grayScal과 equ를 수평으로 붙임
    #cv2.imshow('numpy Histogram Equalization', npImage)

    
    thresholding(img2)
    grad(img2)
    


    """
    # OpenCV를 이용한 방법
    equ = cv2.equalizeHist(img_grayScale)
    # numpy를 이용하여 구현한 것과 동일한 결과를 리턴 함.
    # 하지만 numpy는 컬러 이미지에도 적용가능하지만, cv2.equalizeHist()는 grayscal 이미지만 가능하면 리턴도 grayscal 이미지 임

    res = np.hstack((img_grayScale, equ))
    #  img_grayScal과 equ를 수평으로 붙임

    cv2.imshow('OpenCV Equalizer', res)
    # cv2.imshow('openCV Equalizer', equ)
    """

    cv2.waitKey(0)
    cv2.destroyAllWindows()




img = cv2.imread("gradation01.jpg", cv2.IMREAD_GRAYSCALE)
histogram(img)
#thresholding(img)
#grad(img)



