"""
27강. Image Histogram 배경투사

    이미지 히스토그램 배경투사 : 이미지 분할(Image Segmentation)이나 이미지에서 특정 객체만을 추출하는데 활용

        이미지 분할 : 원하는 이미지 만을 오려내는 작업(아나운서가 파란색 스크린 앞에서 기상예보 등)
        이미지분할 원리
            이미지에서 원하는 객체에 해당하는 픽셀과 확률적으로 비슷한 픽셀들을 구한다.
            얻어진 픽셀 부분을 다른 부분에 비해 좀더 흰색에 가깝도록 만든다.
            원본 이미지와 동일한 크기의 새로운 이미지를 생성한다.
            새롭게 생성한 이미지를 적절하게 Thresholding처리를 거친다
            처리된 이미지와 원본 이미지를 이용해 적절한 연산을 수행하면, 배경 또는 배경을 제외한 나머지 부분을 추출할 수 있다.

        이미지 히스토그램 배경투사 알고리즘
            1. 찾고자 하는 객체의 컬러 히스토그램(M)과 원본 이미지의 컬러 히스토그램(I)을 계산한다.
            2. R = M/I를 구한다. R을 팔레트 값으로 하고 원본 이미지에서 언하는 대상과 확률적으로 일치하는 모든 픽셀을 이용해
                새로운 이미지를 생성. 해당픽셀(x,y)는 B(x,y) = R[h(x,y), s(x,y)]로 구한다.
                h(x,y)는 픽셀 (x,y)의 Hue, s(x,y)는 Saturation. 이렇게 구한 픽셀 집합 B(x,y)의 값을 1과 비교하여
                작은 값을 취한다. B(x,y) = min(B(x,y),1)
            3. 원형 Convolution을 적용 B = D*B, D는 원형 커널
            4. 3번 과정까지 처리하게 되면 픽셀값이 가장 밝은 부분이 원하는 대상이 된다. 적절한 값으로 Thresholding 하여 흰색으로 변환
            5. 4번 과정의 결과와 원본 이미지를 비트연산을 하면 원하는 대상 또는 원하지 않는 대상만을 추출 할 수 있다.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


ix, iy = -1, -1
mode = False
img1, img2 = None, None


#def onMouse()
#   원본 이미지에서 마우스로 사각 영역을 지정하면 해당 영역과 비슷한 부문을 추출하여 화면에 디스플레이 한다.
def onMouse(event, x, y, flag, param):
    global ix, iy, mode, img1, img2

    if event == cv2.EVENT_LBUTTONDOWN:
        mode = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if mode :
            img1 = img2.copy()
            cv2.rectangle(img1, (ix, iy), (x, y), (0,0,255), 2)
            cv2.imshow('original', img1)
    elif event == cv2.EVENT_LBUTTONUP:
        mode = False
        if ix >= x or iy >= y :
            return

        cv2.rectangle(img1, (ix, iy), (x,y), (0,0,255), 2)
        roi = img1[iy:y, ix:x]
        backProjection(img2, roi)

    return

# def backProjection
#   마우스로 지정한 대상과 원본 이미지를 가지고 히스토그램 배경투사 알고리즘을 구현
def backProjection(img, roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    roihist = cv2.calcHist([hsv], [0,1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    #cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    #cv2.normalize() 함수는 인자로 입력된 numpy배열을 정규화 하는 함수
    #   첫번째 인자는 오리지널 배열
    #   두번째 인자는 결과로 나올 배열
    # 예 x = [[0,1,2], [3,4,5]]
    # x = cv2.normalize(x,x,0,255, NORM_MINMAX)의 결과는 x = [[0,51,102], [153,204,255]]가 됨
    # 즉 입력된 x의 최소값은 0이되고 최대갑은 255로 조정후, 나머지 갑들은 비율에 맞춰 재 배열 한다.

    dst = cv2.calcBackProject([hsvt], [0,1], roihist, [0, 180, 0, 256], 1)
    #cv2.calcBackProject() 함수의 인자는 CV2.calcHist() 함수의 인자와 거의 비슷한 구조를 가지고 있음
    # 원본 이미지 컬러 히스토그램을 첫번째 인자로, cv2.calcHist() 함수에서는 None로 입력되었던 부분을
    # 대상부분의 컬러 히스토그램 인자로 입력한 것이 차이 점
    #마지막 인자는 scale인데 , 원본과 같으므로 1로 지정

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cv2.filter2D(dst, -1, disc, dst)
    ret, thr = cv2.threshold(dst, 50, 255, 0)
    thr = cv2.merge((thr, thr, thr))
    res = cv2.bitwise_and(img, thr)

    cv2.imshow('backproject', res)




# def main()
#   원본이미지와 이미지 히스토그램 배경투사가 적용된 결과이미지를 각각 img1과 img2에 담는다.
#   마우스 이벤트 처리를 위해 콜백함수 지정
#   ESC키를 누르면 프로그램 종료
def main(ImgNo):
    global img1, img2

    img1 = cv2.imread(impDef.select_img(ImgNo))
    img2 = img1.copy()

    cv2.namedWindow('original'),cv2.namedWindow('backproject')
    cv2.setMouseCallback('original', onMouse, param = None)

    cv2.imshow('backproject', img2)

    while True:
        cv2.imshow('original', img1)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows( )

#main(23)
main(12)   # 트와이스
#main(5)    # 수지
#main(10)   # Code Image