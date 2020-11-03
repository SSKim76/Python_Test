"""
    16강. 이미지 피라미드
        이미지 피라미드는 2가지 종류가 있습니다.
            가우시안 피라미드(Gaussian Pyramids)
            라플라시안 피라미드(Laplacian Pyramids)

            가우시안 피라미드(cv2.pyrDown() 함수와 cv2.pyrUp() )
            가우시안 피라미드는 상위 단계(해상도가 낮은 이미지) 이미지를 생성하는 것(downsampling; 다운샘플링)과
            하위 단계 이미지(해상도가 높은 이미지)를 생성하는 것(upsampling; 업샘플링)이 있습니다.
            다운샘플링은 하위 단계 이미지의 짝수열, 짝수행에 해당하는 픽셀을 제거함으로써 이미지 해상도를 줄입니다.
            일반적으로 M x N 이미지는 가로 세로가 절반으로 줄어든 M/2 x N/2 해상도 이미지로 변환됩니다.
            업샘플링은 다운샘플링과 반대로 상위 단계 이미지의 짝수열, 짝수행에 픽셀을 추가하여 하위 단계 이미지를 만듭니다.
            하지만 생성된 하위 단계 이미지는 제대로 된 이미지로 나타나지 않고 마치 블러링 효과를 낸 듯한 이미지로 보이게 됩니다.

            라플라시안 피라미드
            라플라시안 피라미드는 가우시안 피라미드 결과로 생성합니다. 원리는 이렇습니다.
            어느 한 이미지(원본 이미지)를 가우시안 피라미드의 상위 단계 이미지2 로 생성합니다.
            이 이미지2 를 가우시안 피라미드의 하위 단계 이미지3 으로 만듭니다.
            원본이미지와 이미지3은 해상도가 같지만(해상도가 다를 수도 있습니다)
            이미지3은 제대로 된 원본이미지로 복구된 것이 아닐 것입니다.
            라플라시안 피라미드의 최하위 단계는 원본이미지에서 이미지3 을 '-' 연산함으로써 생성합니다.
            즉, 원본이미지 - 이미지3 한 결과가 라플라시안 피라미드의 최하위 단계이고,
            그 상위 단계를 같은 방식으로 하면 상위 단계의 라플라시안 피라미드가 생성됩니다.

            피라미드의 활용법으로, 아래의 단계에 따라 코딩을 하면 이미지 블렌딩 효과도 낼 수 있습니다.
                1. 두 개의 이미지를 로드함
                2. 두 개의 이미지에 대한 적절한 단계까지 가우시안 피라미드를 구함
                3. 가우시안 피라미드를 이용해 라플라시안 피라미드를 구함
                4. 각 단계의 라플라시안 피라미드에서 한 이미지의 좌측과 나머지 이미지의 우측을 결합함
                5. 이들을 누적으로 재결합하여 이미지 블렌딩을 완성함
"""

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import default_import as impDef

#from default_import import select_img as selImg
#img = cv2.imread(selImg.select_img(6), cv2.IMREAD_GRAYSCALE)

def pyramid():
    img = cv2.imread(impDef.select_img(5), cv2.IMREAD_GRAYSCALE)
    tmp = img.copy()

    win_titles = ['original', 'Level1', 'Level2', 'Level3']
    g_down = []
    g_up = []

    g_down.append(tmp)

    for i in range(3):
        tmp1 = cv2.pyrDown(tmp)
        g_down.append(tmp1)
        tmp = tmp1

    cv2.imshow('Level 3', tmp)

    for i in range(4):
        cv2.imshow(win_titles[i], g_down[i])

    impDef.close_window()

    for i in range(3):
        tmp = g_down[i+1]
        tmp1 = cv2.pyrUp(tmp)
        g_up.append(tmp1)
        cv2.imshow(win_titles[i], g_up[i])

    impDef.close_window( )

    cv2.imshow('original', img)
    cv2.imshow('Pyramid', g_up[0])

    impDef.close_window()


def pyramid1():
    img = cv2.imread(impDef.select_img(5), cv2.IMREAD_GRAYSCALE)
    tmp = img.copy( )

    win_title = ['Original', 'Level 1', 'Level 2', 'Level 3']
    gDown = []
    gUp = []
    imgShape = []

    gDown.append(tmp)
    imgShape.append(tmp.shape)

    for i in range(3):
        tmp1 = cv2.pyrDown(tmp)
        gDown.append(tmp1)
        imgShape.append(tmp1.shape)
        tmp = tmp1

    for i in range(3):
        tmp = gDown[i+1]
        tmp1 = cv2.pyrUp(tmp)
        tmp = cv2.resize(tmp1, dsize = (imgShape[i][1], imgShape[i][0]), interpolation=cv2.INTER_CUBIC)
        """
        라플라시안 이미지를 만들 때 이용하는 함수 cv2.subtract() 함수는 두 개의 이미지를 '-' 연산하는 함수입니다.
        이 함수의 인자로 들어가는 이미지들은 그 크기가 동일해야 합니다. cv2.subtract() 함수의 두 인자로 사용되는 
        이미지의 크기가 다르면 오류가 발생하고 프로그램은 동작하지 않습니다.
        따라서 위 예제 코드에서는 cv2.subtract() 함수에 들어가는 이미지 크기를 동일하게 맞추기 위해  코드를 강제 삽입한 것입니다.
        """
        gUp.append(tmp)

    for i in range(3):
        tmp = cv2.subtract(gDown[i], gUp[i])
        cv2.imshow(win_title[i], tmp)

    impDef.close_window()

#pyramid()
pyramid1()

