"""
34강. Watershed알고리즘을 이용한 이미지 분할

        cv.distanceTransform()은 이미지의 각 픽셀과 가장 가까운 0인 픽셀과의 거리를 계산하여 리턴.

        label() 함수는 픽셀들이 서로 이어져 있는 부분을 같은 라벨을 붙여 동일한 객체의 일부라는 것을 표시하는 역할.
        label() 함수가 픽셀들이 서로 이어져 있는 것인지 아니면 떨어져 있는 것인지 판단하기 위해 활용되는 구조가 있다.
        기본적인 구조는 검사하려는 픽셀을 기준으로 아래, 위, 좌, 우 4방향의 픽셀들을 검사하는 방법과
        대각선 방향까지 포함하여 8방향의 픽셀들을 검사하는 방법이 있다.
        label() 함수는 위 결과와 라벨링한 개수를 리턴한다.


"""
imort numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label
import default_import as impDef


def waterShed(ImgNo):
    img = cv2.imread(impDef.select_img(ImgNo))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(imgray, 50, 150, apertureSize = 3)
    # Canny Edge Detection결과를 구한다. (GrayScale Image, Threshold Min, Threshold Max)

    # Threshold적용하여 바이너리 이미질 변환
    ret, thr = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    # 노이즈 제거를 위해 Opening을 실행
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations =2)
    #opening = thr


    border = cv2.dilate(opening, kernel, iterations = 3)
    #흰색영역 확장
    border = border - cv2.erode(border, None)
    #흰색영역 확장한 이미지에서 erode 즉 erosion한 이미지를 빼면 동전과 배경의 경계가 나온다.


    dt = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Opening결과 이미지에 거리 변환을 적용
    # cv2.distanceTransform()은 이미지의 각 픽셀과 가장 가까운 0인 픽셀과의 거리를 계산하여 리턴
    dt = ((dt-dt.min()) / (dt.max()-dt.min())*255).astype(np.uint8)


    ret, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    # threshold를 적용, 동전임을 확신할수 있는 부분이며, 마커로 표시할 예정


    marker, ncc = label(dt)
    marker = marker * (255/ncc)

    marker[border==255] = 255
    marker = marker.astype(np.int32)
    cv2.watershed(img, marker)

    marker[marker ==-1] = 0
    marker = marker.astype(np.uint8)
    marker = 255 - marker

    # 마커이미지에서 값이 255(흰색)이 아닌 부분은 모두 검정색으로 처리.
    # 선을 좀 굵게 하기위해 cv2.dalat() 함수를 적용 없어도 상관없음
    #  마커의 경계부분을 빨간색으로 바꾼다.
    marker[marker != 255] = 0
    marker = cv2.dilate(marker, None)
    img[marker==255] = (0, 0, 255)

    cv2.imshow('watershed', img)

    impDef.close_window()

waterShed(34)

