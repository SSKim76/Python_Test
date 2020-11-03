try:
    from numpy import numpy as np
    from cv2 import  cv2
except ImportError:
    pass

# from random import shuffle  # shuffle() 함수 사용을 위해
# import math # sqrt 함수 사용을 위해, sqrt(x) -> 루트 x , 제곱근을 구하는 함수

drawing = False
ix, iy = -1, -1
# 색상값을 위해 사용될 0~255를 멤버로 하는 리스트 생성

#   마우스 이벤트를 처리할 콜백 함수
#   cv2.setMouseCallBack()함수의 인자로 지정되어 호출
#   event : 마우스 이벤트
#   x, y: 마우스 이벤트가 일어난 위치
#   flags: 여기서는 사용하지 않음
#   param: cv2.setMouseCallback() 함수에서 전달받은 사용자 데이터


def onMouse(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if ix >= x:
            k = ix
            ix = x
            x = k

        if iy >= y:
            k = iy
            iy = y
            y = k

    #subimg = img[300:400, 320:720]
    # 원본이미지의 Y축 300 ~ 400, X축 320 ~ 720 -> 400 x 100 크기의 이미지
        subimg = param[iy:y, ix:x]
        cv2.imshow('cut img', subimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows( )





def mouseBrush():

    global mode

    imgfile = 'img/100.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    cv2.namedWindow('paint')
    cv2.setMouseCallback('paint', onMouse, param = img)


    while True:
        cv2.imshow('paint', img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27 :
            break

    cv2.destroyAllWindows()

mouseBrush()

