"""
33강. 허프변환을 이용한 원찾기

    cv2.HoughCircles(imgray,cv2.HOUGH_GRADIENT,1,10, param1=60, param2=50, minRadius=0, maxRadius=0)
       cv2.HOUGH_GRADIENT : 허프변환을 이용하여 원을 찾는 방법, 현재는 이 방법 하나 밖에 없음
       1 : 원본 이미지와 허프변환 카운팅 결과 이미지의 비(ratio), 걍 1로 설정
       10 : 찾은 원들의 중심간의 최소 거리. 중심간의 거리가 이 값보다 작으면 나중에 찾은 원은 무시한다.
            이 값을 적절하게 조정하여 제대로 된 원을 찾도록 한다. 이 값을 0으로 설정하면 오류 발생 즉 동심원은 동시에 찾을 수가 없다.
       param1 = 60 : cv2.HoughCircles() 함수는 내부적으로 Canny Edge Detection을 사용하는데, Canny()함수의 인자로 들어가는
                    maxVal의 값. 이 값을 적절히 조절하여 제대로 된 원을 찾도록 한다.
       param2 = 50 : 원으로 판단하는 허프변환 카운팅 값. 너무 작으면 원하지 않은 많은 원들이 찾아지고, 너무 크면 원을 못 찾을 수 있음
       minRadius=0 : 검출하려는 원의 최소 반지름. 크기를 알 수 없는 경우 0으로 지정
       maxRadius=0 : 컴출하려는 원의 최대 반지름. 크기를 알 수 없는 경우 0으로 지정. 음수를 지정하면 원의 중심만 리턴
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef



def houghCircle(ImgNo, param1, param2):
    img = cv2.imread(impDef.select_img(ImgNo))
    img2 = img.copy()

    img2 = cv2.GaussianBlur(img2, (3,3), 0)
    # 옵션 - 가우시안 필터 적용, 원본 이미지에서 원 찾기가 잘 되지 않을 경우, 적절한 커널값으로 이디안 필터나
    # 가우시안 필터를 적용하면 더 좋은 결과를 얻을 수 있다.

    imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 10, param1 = param1, param2 = param2, minRadius = 0, maxRadius = 0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Numpy.around() : circles의 값들을 반올림/반내림하고 이를 uint16으로 변환

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            #원 외곽선 그리기
            cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 0), 2)
            #원 중심 그리기
            cv2.circle(img, (i[0], i[1]), 2, (255, 255, 255), 3)

            cv2.imshow('HoughCircles', img)
            impDef.close_window()
    else:
        print('원을 찾지 못했습니다.')

    cv2.imshow('res', img)

houghCircle(33, 60, 50)