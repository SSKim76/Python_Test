"""
#TODO

TrackBar Test

    cv2.createTrackbar(trackbarname, windowname, start, end, onChange):
        trackbarname : 트랙바의 이름
        windowname : 트랙바가 생성될 윈도우 이름
        start : 트랙바 시작 값
        end : 트랩바 끝 값
        onChange : 트랙바 이벤트 발생시 수행되는 콜백 함수

    cv2.getTrackbarPos(trackbarname, windowname)
        트랙바의 현재 위치를 리턴하는 함수
        trackbarname : 트랙바 이름
        windowname : 트랙바가 생성된 윈도우 이름

        # : Slice  -> start:stop[:step] 형식으로 사용가능, [:step] 써도 되고 안써도 되고
        # a[start:end]  -> start부터 end-1까지의 item
        # a[start:] -> start부터 리스트 끝까지의 item
        # a[:end] -> 처음부터 end-1까지의 item
        # a[:]  -> 리스트의 모든 item

        # a[-1] -> 맨 뒤의 item
        # a[-2:] -> 맨 뒤에서 부터 item 2개
        # a[:-n] -> 맨 뒤의 item n개 빼고 전부
"""

import numpy as np
import cv2

def onChange(x):
    pass
    # 트랙바 변경시 아무작업도 하지 않음

def trackbar():
    img = np.zeros((200, 512, 3), np.uint8)
    # 200 x 512 크기의 검정색 그림판을 생성
    cv2.namedWindow('Color_Palette')
    # "Color_Palette"라는 이름의 윈도우 생성

    cv2.createTrackbar('B', 'Color_Palette', 0, 255, onChange)
    cv2.createTrackbar('G', 'Color_Palette', 0, 255, onChange)
    cv2.createTrackbar('R', 'Color_Palette', 0, 255, onChange)
    # Color_Palette윈도우에 0 ~ 255까지 변경가능한 트랙바 B, G, R을 생성


    switch = '0:OFF \n 1:ON'
    cv2.createTrackbar(switch, 'Color_Palette', 0, 1, onChange)
    # On/Off 스위치 역할을 할 트랙바 생성

    while True :
        cv2.imshow('Color_Palette', img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

         # 각 트랙바의 현재 값을 읽어 온다.
        b = cv2.getTrackbarPos('B', 'Color_Palette')
        g = cv2.getTrackbarPos('G', 'Color_Palette')
        r = cv2.getTrackbarPos('R', 'Color_Palette')
        s = cv2.getTrackbarPos('switch', 'Color_Palette')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

    cv2.destroyAllWindows()

trackbar()





