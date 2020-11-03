"""
35강. GrabCut을 이용한 전경추출

    GrabCut
        사용자가 전경과 배경이 되는 부분을 마킹함으로써 보다 나은 결과를 만들어 주는 방법

    GrabCut 절차
        1. 추출하고자 하는 전경을 영역으로 잡아준다.
        2. GrabCut 알고리즘 적용
        3. 남아있는 배경을 마킹
        4. 마킹한 부분만 따로 떼어낸 이미지를 마스크로 만들어 GrabCut 알고리즘 적용
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


BLUE, GREEN, RED, BLACK, WHITE = (255,0,0), (0,255,0), (0,0,255), (0,0,0), (255,255,255)
DRAW_BG = {'color':BLACK, 'val':0}
DRAW_FG = {'color':WHITE, 'val':1}

rect = (0,0,1,1)
drawing = False
rectangle = False
rect_over = False
rect_or_mask = 100
value = DRAW_FG
thickness = 3


def onMouse(event, x, y, flags, param):
    global ix, iy, img, img2, drawing, value, mask, rectangle
    global rect, rect_or_mask, rect_over

    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event ==cv2.EVENT_MOUSEMOVE:
        if rectangle:
            img = img2.copy()
            cv2.rectangle(img, (ix, iy), (x, y), RED, 2)
            rect = (min(ix,x), min(iy,y), abs(ix-x), abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img, (ix, iy), (x, y), RED, 2)
        rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        rect_or_mask = 0
        print('n:적용하기')

    if event == cv2.EVENT_LBUTTONDOWN:
        if not rect_over:
            print('마우스 왼쪽 버튼을 누른채로 전경이 되는 부분을 선택하세요')
        else:
            drawing = True
            cv2.circle(img, (x,y), thickness, value['color'], -1)
            cv2.circle(mask, (x,y), thickness, value['val'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x,y), thickness, value['color'], -1)
            cv2.circle(mask, (x,y), thickness, value['val'], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    return



def grabcut(ImgNo):
    global ix, iy, img, img2, drawing, value, mask, rectangle
    global rect, rect_or_mask, rect_over

    img = cv2.imread(impDef.select_img(ImgNo))
    img2 = img.copy()

    mask = np.zeros(img.shape[:2], dtype = np.uint8)
    output = np.zeros(img.shape, np.uint8)

    cv2.namedWindow('input')
    cv2.namedWindow('output')
    cv2.setMouseCallback('input', onMouse, param = (img, img2))
    cv2.moveWindow('input', img.shape[1]+10, 90)
    print('오른쪽 마우스 버튼을 누르고 영역을 지정한 후 n을 누르세요')

    while True:
        cv2.imshow('output', output)
        cv2.imshow('input', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if k == ord('0'):
            print('왼쪽 마우스로 제거할 부분을 표시한 후, n을 누르세요')
            value = DRAW_BG
        elif k == ord('1'):
            print('왼쪽 마우스로 복원할 부분을 표시한 후, n을 누르세요')
            value = DRAW_FG
        elif k == ord('r'):
            print('Reset.....')
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype = np.uint8)
            output = np.zeros(img.shape, np.uint8)
            print('0:제거배경선택    1:복원전경선택    n:적용    r:reset')
        elif k == ord('n'):
            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)

            if rect_or_mask == 0:
                cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:
                cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

            print('0:제거배경선택    1:복원전경선택    n:적용    r:reset')

        mask2 = np.where((mask==1)+(mask==3), 255,0).astype('uint8')
        output = cv2.bitwise_and(img2, img2, mask=mask2)

    impDef.close_window()

grabcut(35)









