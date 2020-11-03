"""
    09. 색공간 바꾸기 및 색 추적

        1. BGR 색공간을 Gray로 변경하거나, HSV로 변경하기
        2. 비디오 프레임에서 특정한 색만 추출하여 추적하기

        cv2.cvtColor()
        BGR 색공간으로 생성한 Color를 HSV 값으로 전환한

        cv2.inRange()
        소스인 hsv의 모든 값을 lower_blue, upper_blue로 지정한 범위에 있는지 체크한 후,
        범위에 해당하는 부분은 그 값 그대로, 나머지 부분은 0으로 채워서 결과값을 반환합니다.



        OpenCV는 150가지 이상의 색공간 변경 메쏘드를 제공하고 있습니다.
        하지만 우리는 가장 많이 사용되는 BGR - GRAY, BGR - HSV 색공간 변경 방법만
        다루어 보도록 하겠습니다.

        BGR은 Blue, Green, Red 값으로 하나의 색을 결정하는 것이죠.
        HSV는 Hue(색상), Saturation(채도), Value(진하기)로 색을 결정합니다
        OpenCV에서는 Hue의 범위를 [0, 179],
        Saturation과 Value의 범위를 [0, 255]로 정의하고 있습니다.

"""

import numpy as np
import cv2 as cv2

def hsv():
    blue = np.uint8([[[255, 0, 0]]])
    # Blue 픽셀 1개에 해당하는 numpy array를 생성합니다
    green = np.uint8([[[0, 255, 0]]])
    red = np.uint8([[[0, 0, 255]]])

    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    # BGR 색공간으로 생성한 Blue를 HSV 값으로 전환한 것을 hsv_blue에 담습니다
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

    print('hsv for blue : ', hsv_blue)
    print('hsv for green : ', hsv_green)
    print('hsv for red : ', hsv_red)

# hsv()
# 결과
# HSV for Blue: (120, 255, 255)
# HSV for Green: (60, 255, 255)
# HSV for Red: (0, 255, 255)

def tracking():
    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture(0)
    except:
        print('카메라 구동실패!!')
        return

    while True:
        ret, frame = cap.read()

        # BGR을 HSV모드로 변환
        hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV에서 각 B, G, R로 정할 범위 설정
        lower_blue = np.array([110, 100, 100])
        upper_blue = np.array([130, 255, 255])

        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])

        lower_red = np.array([-10, 100, 100])
        upper_red = np.array([10, 255, 255])

        # HSV 이미지에서 각 R, G, B만 추축하기 위한 임계값
        mask_blue = cv2.inRange(hsv1, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv1, lower_green, upper_green)
        mask_red = cv2.inRange(hsv1, lower_red, upper_red)

        # Mask와 원본이미지를 비트 연산
        res1 = cv2.bitwise_and(frame, frame, mask = mask_blue)
        res2 = cv2.bitwise_and(frame, frame, mask = mask_green)
        res3 = cv2.bitwise_and(frame, frame, mask = mask_red)

        cv2.imshow('ORIGINAL', frame)
        cv2.imshow('BLUE', res1)
        cv2.imshow('GREEN', res2)
        cv2.imshow('RED', res3)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

tracking()





