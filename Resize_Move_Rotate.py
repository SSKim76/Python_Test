"""
    11강 이미지 변환 - 리사이징, 이동, 회전, 원근효과

        cv2.resize(img, None, fx, fy, interpllation = cv2.INTER_AREA)
            img : 원본이미지
            None : dsize를 나타내는 튜플값(가로픽셀 수, 세로픽셀 수)
            fx, fy : 각각 가로, 세로방향으로의 배율 인자(0.5 = 원래크기의 0.5배)
            interplolation = : 리사이징을 수행할 때 적용할 Interpolation 방법
                cv2.INTER_NEAREST :
                cv2.INTER_LINEAR : bilinea(디폴드 값)
                cv2.INTER_AREA : 픽셀 영역관계를 이용한 resampling 방법, 축소에 선호
                cv2.INTER_CUBIC : 4x4 pixel에 적용되는 bicubic interpolation
                cv2.INTER_LANCZOS4 : 8x8 Pixel에 적용된는 lanczos interpolation
                일반적으로 축소는 cv2.INTER_AREA
                확대는 cv2.INTER_CUBIC + cv2.INTER_LINEAR을 사용

        cv2.warpAffine() -> 이미지 이동(모든 픽셀을 X, Y 방향을로 일정량 선형으로 Shift)
            이미지의 한 픽셀을 (tx, ty) 만큼 이동을 나타내는 행렬은 아래와 같습니다.
            위 행렬 M의 원소는 각각 M11, M12, M13 = (1, 0, tx) 이고, M21, M22, M23 = (0, 1, ty) 입니다.


        cv2.getPerspectiveTransform()
            먼저 원근감을 주는 변환 매트릭스를 구하는 함수
            이 함수는 4개의 좌표로 된 인자가 필요합니다.
            변환되기 전 4개의 좌표와 변환된 후 4개의 좌표입니다.

            원근감 변환 매트릭스 M을 구하면 cv2.warpPerspective() 함수의 인자로 입력하여 결과를 얻습니다.
            매트릭스 M은 3x3 크기의 매트릭스여야 하며, cv2.warpPerspective(img, M, (cols, rows)) 함수는 아래 식의 계산을 수행합니다.
            img 의 모든 점 (x, y)는 변환 후 dst(x, y)가 됩니다.

            Perspective 변환을 잘 이용하면 아래와 같이 기울어져 있는 그림을 바로 세울 수 있다
"""


import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

#img_url = 'img/images_1.jpg'
img_url = 'img/threshold_original.png'      # 여자이미지
#img_url = 'img/tile.jpg'                            # 타일이미지
#img_url = 'img/mulit_light.jpg'
#img_url = 'img/sdoku.jpg'                      #sdoku


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
    }.get(sel_num,'img/girl.png')


"""
    # img_url = 'img/images_1.jpg'
    img_url = 'img/threshold_original.png'          # 여자이미지
    # img_url = 'img/tile.jpg'                            # 타일이미지
    # img_url = 'img/mulit_light.jpg'
    return img_url
"""


def close_window():
    cv2.waitKey(0)
    cv2.destroyAllWindows( )



def transform():
    img = cv2.imread(img_url)
    h, w = img.shape[:2]

    img2 = cv2.resize(img, None, fx =0.5, fy = 1, interpolation = cv2.INTER_AREA)
    img3 = cv2.resize(img, None, fx = 1, fy = 0.5, interpolation = cv2.INTER_AREA)
    img4 = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

    cv2.imshow("Original", img)
    cv2.imshow('fx = 0.5', img2)
    cv2.imshow('fy = 0.5', img3)
    cv2.imshow('fx, fy = 0.5', img4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def transform_move():
    img = cv2.imread(img_url)
    h, w = img.shape[:2]

    fx = 100    # X축 이동거리
    fy = 50     # Y축 이동거리

    M = np.float32([[1, 0, fx], [0, 1, fy]])
    img2 = cv2.warpAffine(img, M, (w, h))
    """
        cv2.warpAffine() 함수는 img의 모든 좌표 (a, b)를 
        (M11 x a + M12 x b + M13, M21 x a + M22 x b + M23) 으로 변환합니다.
            img: 변환할 소스 이미지
            M: 2 x 3 변환 매트릭스
            (w, h): 출력될 이미지 사이즈
    """

    cv2.imshow('original', img)
    cv2.imshow('shift image', img2)

    close_window()




def transform_rotate():
    img_url = select_img()
    img = cv2.imread(img_url)
    h, w = img.shape[:2]

    angle45 = 45
    angle90 = 90

    # 원본 이미지 무게중심을 회전 중심으로 하고 45, 90도, scale=1인 회전 변환 매트릭스 구합니다
    M1 = cv2.getRotationMatrix2D((w/2, h/2), angle45, 1)
    M2 = cv2.getRotationMatrix2D((w/2, h/2), angle90, 1)

    img2 = cv2.warpAffine(img, M1, (w, h))
    img3 = cv2.warpAffine(img, M2, (w, h))

    cv2.imshow('original', img)
    cv2.imshow('45-Rotated', img2)
    cv2.imshow('90-Rotated', img3)

    close_window()


def transform_perspective():
    img_url = select_img(8)
    img = cv2.imread(img_url)
    h, w = img.shape[:2]

    pts1 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts2 = np.float32([[56, 65], [368, 52], [28,387], [389, 390]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img2 = cv2.warpPerspective(img, M, (w, h))

    cv2.imshow('original', img)
    cv2.imshow('Perspective', img2)

    close_window()


# transform()               # Resize
# transform_move()       # Move
# transform_rotate()      # Rotate
transform_perspective()





