"""
    10강  Thresholding

        OpenCV 이미지 프로세싱에서 thresholding을 적용하려면 grayscale 이미지로 변환하여 적용

        Global Thresholding
        OpenCV에서 하나의 이미지에 전역으로 적용될 하나의 문턱값을 이용해 thresholding 기능을
        제공하는 함수가 있습니다.

        cv2.threshold(img, threshold_value, value, flag)
            img: Grayscale 이미지
            threshold_value: 픽셀 문턱값
            value: 픽셀 문턱값보다 클 때 적용되는 최대값(적용되는 플래그에 따라 픽셀 문턱값보다 작을 때 적용되는 최대값)
            flag: 문턱값 적용 방법 또는 스타일
                cv2.THRESH_BINARY: 픽셀 값이 threshold_value 보다 크면 value, 작으면 0으로 할당
                cv2.THRESH_BINARY_INV: 픽셀 값이 threshold_value 보다 크면 0, 작으면 value로 할당
                cv2.THRESH_TRUNC: 픽셀 값이 threshold_value 보다 크면 threshold_value, 작으면 픽셀 값 그대로 할당
                cv2.THRESH_TOERO: 픽셀 값이 threshold_value 보다 크면 픽셀 값 그대로, 작으면 0으로 할당
                cv2.THRESH_TOZERO_INV: 픽셀 값이 threshold_value 보다 크면 0, 작으면 픽셀 값 그대로 할당

        Adaptive Thresholding
            앞서 설명한 cv2.threshold()함수를 이용한 global thresholding 방법은 이미지 전반에 걸쳐 적용되는
            하나의 문턱값을 활용한 로직을 사용했습니다. 이런 방법은 이미지의 각부분의 광원 조건에 따라
            이미지 프로세싱을 함에 있어 효과적인 방법이 아닐 수도 있습니다.

            Adaptive Thresholding은 이미지의 서로 다른 작은 영역에 적용되는 문턱값을 계산하고
            이를 이미지에 적용함으로써 보다 나은 결과를 도출하는데 사용되는 방법입니다.
            이 기능을 제공하는 함수는 cv2.adaptiveThreshold() 함수이며, 함수의 인자는 아래와 같습니다.

            cv2.adaptiveThreshold(img, value, adaptiveMethod, thresholdType, blocksize, C)
                img: Grayscale 이미지
                value: adaptiveMethod에 의해 계산된 문턱값과 thresholdType에 의해 픽셀에 적용될 최대값
                adaptiveMethod: 사용할 Adaptive Thresholding 알고리즘
                    cv2.ADAPTIVE_THRESH_MEAN_C: 적용할 픽셀 (x,y)를 중심으로 하는 blocksize x blocksize 안에 있는 픽셀값의 평균에서 C를 뺀 값을 문턱값으로 함
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 적용할 픽셀 (x,y)를 중심으로 하는 blocksize x blocksize안에 있는 Gaussian 윈도우 기반 가중치들의 합에서 C를 뺀 값을 문턱값으로 함
                blocksize: 픽셀에 적용할 문턱값을 계산하기 위한 블럭 크기. 적용될 픽셀이 블럭의 중심이 됨. 따라서 blocksize는 홀수여야 함
                C: 보정 상수로, 이 값이 양수이면 계산된 adaptive 문턱값에서 빼고, 음수면 더해줌. 0이면 그대로..


        Otsu's Binarization
            global thresholding 방법에서 문턱값으로 우리가 정한 임의의 값을 사용했습니다.
            그런데 어떤 이미지에 적용되어 가장 좋은 결과를 내놓게될 문턱값은 어떻게 얻을 수 있을까요?
            정답은 시행착오를 거치는 방법입니다.

            만약 이미지 히스토그램이 두개의 봉우리를 가지는 bimodal 이미지라고 하면
            이 이미지에 대한 문턱값으로 두 봉우리 사이의 값을 취하면 가장 좋은 결과를 얻을 수 있습니다.

            Otsu Binarization은 이미지 히스토그램을 분석한 후 중간값을 취하여 thresholding 합니다.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


#threshold_original.jpg
#threshold_original.png

#img_url = 'img/images_1.jpg'
#img_url = 'img/threshold_original.png'      #여자이미지


#img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
threshold = 150
value = 255

"""
# ret : 임계값 저장, thr = 변환 이미지 저장
ret, thr1 = cv2.threshold(img, threshold, value, cv2.THRESH_BINARY)
ret, thr2 = cv2.threshold(img, threshold, value, cv2.THRESH_BINARY_INV)
ret, thr3 = cv2.threshold(img, threshold, value, cv2.THRESH_MASK)
ret, thr4 = cv2.threshold(img, threshold, value, cv2.THRESH_OTSU)
ret, thr5 = cv2.threshold(img, threshold, value, cv2.THRESH_TOZERO)
ret, thr6 = cv2.threshold(img, threshold, value, cv2.THRESH_TOZERO_INV)
ret, thr7 = cv2.threshold(img, threshold, value, cv2.THRESH_TRIANGLE)
ret, thr8 = cv2.threshold(img, threshold, value, cv2.THRESH_TRUNC)

cv2.imshow('original', img)
cv2.imshow('BINARY', thr1)
cv2.imshow('BINARY_INV', thr2)
cv2.imshow('MASK', thr3)
cv2.imshow('OTSU', thr4)
cv2.imshow('TOZERO', thr5)
cv2.imshow('TOZERO_INV', thr6)
cv2.imshow('TRIANGLE', thr7)
cv2.imshow('TRUNC', thr8)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""


def thresholding(ImgNo, threshold, value):

    img = cv2.imread(impDef.select_img(ImgNo), cv2.IMREAD_GRAYSCALE)

    ret, thr9 = cv2.threshold(img, threshold, value, cv2.THRESH_BINARY)
    thr10 = cv2.adaptiveThreshold(img, value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thr11 = cv2.adaptiveThreshold(img, value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    title = ['original', 'Global Thresholding(v=100)', 'Adaptive MEAN', 'Adaptive GAUSSIAN']
    images = [img, thr9, thr10, thr11]

    for i in range(4):
        cv2.imshow(title[i], images[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()



"""
    >>> test_list = ['one', 'two', 'three'] 
    >>> for i in test_list: 
    ...     print(i)
    ... 
    one 
    two 
    three
    ['one', 'two', 'three'] 리스트의 첫 번째 요소인 'one'이 먼저 i 변수에 대입된 후 
    print(i) 문장을 수행한다. 다음에 두 번째 요소 'two'가 i 변수에 대입된 후 
    print(i) 문장을 수행하고 리스트의 마지막 요소까지 이것을 반복한다.
"""

def thresholding2(ImgNo, threshold, value):


    img = cv2.imread(impDef.select_img(ImgNo), cv2.IMREAD_GRAYSCALE)

    # 전역 Thresholdt 적용
    ret, thr12 = cv2.threshold(img, threshold, value, cv2.THRESH_BINARY)

    # Oust 바이너리제이션
    ret, thr13 = cv2.threshold(img, 0, value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #Otsu Binarization을 적용하는 함수는 따로 없고,
    # cv2.threshold() 함수에 cv2.THRESH_OTSU 플래그 값을 thresholding 플래그에 더하고
    # 문턱값으로 0을 전달해주면 됩니다.
    # 이렇게 하면 cv2.threshold() 함수는 적절한 문턱값을 계산한 후 이를 적용한 결과를 리턴합니다.


    # 가우시안블러 적용 후, Otsu 바이너리제이션
    blur = cv2.GaussianBlur(img, (5,5), 0)
    ret, thr14 = cv2.threshold(blur, 0, value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = ['original noisy', 'Histogram', 'G-Thresholding',
                'original noisy', 'Histogram', 'Otsu Thresholding',
                'Gaussian-Filter', 'Histogram', 'Otsu Thresholding']
    images = [img, 0, thr12, img, 0, thr13, img, 0, thr14]

    for i in range(3):
        plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

        plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        #plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray)')
        #plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

    plt.show()


thresholding(100, 200, 255)
#thresholding2(100, 100, 255)
