"""
18. Image Contour 응용1

   이번 강좌에서는 폐곡선인 Contour의 면적(area), contour의 둘레 길이(perimeter),
   폐곡선인 Contour의 중심(centeroid) 등과 같이 우리가 살펴보고자 하는 contour의
   특성이 어떤지를 살펴보도록 하겠습니다.

    이미지 모멘트(Image Moments)
        이미지 모멘트는 객체의 무게중심, 객체의 면적 등과 같은 특성을 계산할 때 유용합니다.
        OpenCV의 cv2.moments() 함수는 이미지 모멘트를 계산하고 이를 사전형 자료에 담아 리턴합니다.
        모멘트 종류는 3가지이며 아래와 같이 총 24개의 값을 가집니다.

        공간 모멘트(Spatial Moments)
            m00, m10, m01, m20, m11, m02, m30, m21, m12, m03

        중심 모멘트(Central Moments)
            mu20, mu11, mu02, mu30, mu21, mu12, mu03

        평준화된 중심 모멘트(Central Normalized Moments)
            nu20, nu11, nu02, nu30, nu21, nu03
        이미지 모멘트에 대한 자세한 내용은 이 강좌의 범위를 넘어서는 내용이므로 생략하겠습니다.

    cv2.moments() 함수의 인자는 1개로, 1xN 또는 Nx1 크기의 Numpy array 입니다.
    cv2.findContours() 함수는 이미지에서 contour를 찾은 후 리스트형 자료에 담아 리턴합니다.
    하나의 Contour는 1xN 크기의 Numpy Array 입니다.
    따라서 cv2.moments() 함수는 다음과 같이 활용하면 됩니다.
    cv2.findContours() 함수로 이미지 contour들을 찾음.
    찾은 contour에서 이미지 모멘트를 구하고자 하는 contour 1개를 정함
    이 contour를 cv2.moments() 함수의 인자로 전달하여 이미지 모멘트를 구함

    >>> contour = cv2.contours[0]
    >>> mmt = cv2.moments(contour)
    cv2.findContour() 함수로 찾은 contour들 중, 첫번째 contour에 대한 이미지 모멘트를 구합니다.
    그런 후, 계산된 이미지 모멘트를 화면에 출력해보면 이미지 모멘트 종류에 따른 여러가지
    모멘트 값들이 화면에 출력될 겁니다. 물론 위에서 말한대로 총 24개의 값이죠.
    여기서 우리가 기억해야 할 것은 contour의 무게중심을 구하는 방법입니다.
    >>> cx = int(mmt['m10']/mmt['m00'])
    >>> cy = int(mmt['m01']/mmt['m00'])

    Contour Area와 Contour Perimeter
        Contour Area는 폐곡선 형태의 Contour로 둘러싸인 부분의 면적을 의미하며
        Contour Perimeter는 Contour 호의 길이를 의미합니다.

        OpenCV에서는 이 두 가지 기능을 제공하는 함수는 각각 cv2.contourArea(), cv2.arcLength() 함수입니다.
        참고로 cv2.moments() 함수로 얻은 사전자료에서 키값이 'm00'인 값이 Contour Area의 값입니다.
        즉, Contour Area = cv2.contourArea(contour) = mmt['m00']

        Contour 둘레 길이를 리턴하는 함수 cv2.arcLength()는 2개의 인자를 가지는데
        두 번째 인자는 Contour가 폐곡선인지 아니면 양끝이 열려 있는 호인지를 나타내는 boolean 값 입니다.
        이 값이 True이면 Contour가 폐곡선임을, False이면 Contour가 열려있는 호임을 나타냅니다.

    Contour 근사법 (Contour Approximation)
        이미지 프로세싱을 하다보면 도형 외곽선을 꼭지점수가 원래 보다 적은 다른 모양으로
        바꿀 필요성이 생길 때가 있습니다. 이는 Douglas-Peucker 알고리즘을 적용하여
        꼭지점 줄이기 근사를 수행합니다.

        도형이 하나의 색상으로 채워져 있을 경우, 이 도형의 외곽선은 Contour로 나타낼 수 있죠.
        아래와 같은 군데 군데 찢겨나간 직사각형 그림이 있습니다.
        이렇게 찢겨진 사각형을 복원한다고 할 때, Contour 근사법은 훌륭한 도구가 될 수 있습니다.
        위 그림에서 보이는 찢어진 사각형은 30개의 꼭지점을 가지고 있습니다.
        이 이미지의 Contour를 그려보면 아래 그림과 같이 나옵니다.

        epsilon의 값이 작으면 오리지널 Contour와 비슷한 결과가 도출되고(물론, 꼭지점의 개수는 줄어 있습니다.)
        epsilon의 값이 크면 오리지널 Contour와 차이가 있는 결과가 나옵니다.
        만약 epsilon이 더 커지게 되면 꼭지점의 개수가 0개인 점으로 결과가 나올 수 있습니다.
"""

import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import default_import as impDef

def moment():



    ret , thr = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0]
    mmt = cv2.moments(contour)

    for key, val in mmt.items():
        print('%s:\t%.5f' %(key, val))

    #cx = int(mmt['m10']/mmt['m00'])
    #cy = int(mmt['m01']/mmt['m00'])
    #print(cx, cy)


def contour():
    ret , thr = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    cv2.drawContours(img, [cnt], 0, (255, 255, 0), 1)

    print('contour 면적 : ', area)
    print('contour 길이 : ', perimeter)

    cv2.imshow('contour', img)

    impDef.close_window()


def contour1():
    ret, thr = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('threshold', thr)

    cnt = contours[0]
    cv2.drawContours(img, [cnt], 0, (0, 0, 0), 1)

    epsilon1 = 0.01 * cv2.arcLength(cnt, True)
    epsilon2 = 0.1 * cv2.arcLength(cnt, True)

    approx1 = cv2.approxPolyDP(cnt, epsilon1, True)
    approx2 = cv2.approxPolyDP(cnt, epsilon2, True)
    """
        cv2.approxPolyDP() 함수는 인자로 주어진 곡선 또는 다각형을 epsilon 값에 따라 
        꼭지점수를 줄여 새로운 곡선이나 다각형을 생성하여 리턴합니다.
        
        cv2.approxPolyDP() 함수는 세 개의 인자를 가집니다.
            cnt: Numpy Array 형식의 곡선 또는 다각형. 우리의 예에서는 Contour를 입력함
            epsilon: 근사 정확도를 위한 값. 이 값은 오리지널 커브와 근사 커브간 거리의 최대값으로 사용
            True: 세 번째 인자가 True이면 폐곡선, False이면 양끝이 열려있는 곡선임을 의미

            여기서 중요한 값이 epsilon 입니다. epsilon의 크기에 따라 근사되는 결과가 다르게 나옵니다.
            우리의 예제에서는 근사값으로 epsilon1, epsilon2를 사용했는데, 
            코드를 보시면 cv2.arcLength() 함수의 리턴값에 일정한 숫자를 곱한 값을 이용했습니다.
            즉, 오리지널 Contour의 둘레 길이의 1%를 epsilon1, 둘레 길이의 10%를 epsilon2 값으로 할당헀죠~
        """


    cv2.drawContours(img1, [approx1], 0, (0, 255, 0), 3)
    cv2.drawContours(img2, [approx2], 0, (0, 255, 0), 3)

    cv2.imshow('contour', img)
    cv2.imshow('Approx1', img1)
    cv2.imshow('Approx2', img2)

    impDef.close_window()

#imgray = cv2.imread(impDef.select_img(18), cv2.COLOR_BGR2GRAY)
img = cv2.imread(impDef.select_img(18), cv2.COLOR_BGR2GRAY)

#img = cv2.imread(impDef.select_img(18))
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = imgray.copy()
img2 = imgray.copy()

moment()
contour()
contour1()


