"""
17강. Image Contour
    이미지 Contour
        Contour란 같은 값을 가진 곳을 연결한 선이라고 생각하면 됩니다.
        우리 주위에 자주 접할 수 있는 Contour의 예로 지도에서 같은 높이를 가진 지점을 연결한
        등고선, 기상도에서 같은 기압을 가진 곳을 연결한 등압선 등이 있죠.

        이미지 Contour란 동일한 색 또는 동일한 색상 강도(Color Intensity)를 가진 부분의 가장
        자리 경계를 연결한 선입니다.
        이미지 Contour는 이미지에 있는 물체의 모양 분석이나 객체 인식 등에 유용하게 활용되는 도구입니다.
        보다 정확한 이미지 Contour를 확보하기 위해 바이너리 이미지를 사용합니다.
        즉, 이미지에서 Contour를 찾기 전에 우리가 이미 배웠던 threshold나
        canny edge detection을 적용하는 것이 좋습니다.

        OpenCV의 cv2.findContours() 함수는 Suzuki85라는 알고리즘을 이용해서
        이미지에서 Contour를 찾는 함수입니다.
        이 함수는 원본 이미지를 변경시키기 때문에 향후 원본이미지를 활용하기 위해서는
        원본 이미지의 복사본을 가지고 Contour를 찾도록 하세요~

        주의!! OpenCV에서 Contour 찾기는 검정색 배경에서 흰색 물체를 찾는 것과 비슷합니다.
        따라서 Contour를 찾고자 하는 대상은 흰색으로, 배경은 검정색으로 변경해야 함을 꼭 기억하세요!!


        cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.findContours() 함수는 thr, 이미지에서 찾은 contour와 contour들 간의 계층 구조를 리턴합니다.
            그런데 우리는 찾은 contour만 관심이 있으므로 리턴값 3개 중 필요없는 녀석들은 '_' 로 리턴 받았습니다.
            이 함수는 세 개의 인자를 가지고 있습니다.

        thresh: contour 찾기를 할 소스 이미지. thresholding을 통해 변환된 바이너리 이미지어야 함
        cv2.RETR_TREE: 두 번째 인자는 contour 추출 모드이며, 2번째 리턴값인 hierarchy의 값에 영향을 줌
        cv2.RETR_EXTERNAL: 이미지의 가장 바깥쪽의 contour만 추출
        cv2.RETR_LIST: contour 간 계층구조 상관관계를 고려하지 않고 contour를 추출
        cv2.RETR_CCOMP: 이미지에서 모든 contour를 추출한 후, 2단계 contour 계층 구조로 구성함.
            1단계 계층에서는 외곽 경계 부분을, 2단계 계층에서는 구멍(hole)의 경계 부분을 나타내는 contour로 구성됨
        cv2.RETR_TREE: 이미지에서 모든 contour를 추출하고 Contour들간의 상관관계를 추출함

        cv2.CHAIN_APPROX_SIMPLE: 세 번째 인자는 contour 근사 방법임
        cv2.CHAIN_APPROX_NONE: contour를 구성하는 모든 점을 저장함.
        cv2.CHAIN_APPROX_SIMPLE: contour의 수평, 수직, 대각선 방향의 점은 모두 버리고 끝 점만 남겨둠.
            예를 들어 똑바로 세워진 직사각형의 경우, 4개 모서리점만 남기고 다 버림
        cv2.CHAIN_APPROX_TC89__1: Teh-Chin 연결 근사 알고리즘(Teh-Chin chain approximation algorithm)을 적용함



"""
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import default_import as impDef


def contour():
    img = cv2.imread(impDef.select_img(10), cv2.COLOR_BGR2GRAY)
    #img = cv2.imread('img/Code_Sample.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img1 = img.copy()

    #threshold_level = list(range(100, 200, 5))
    #for i in threshold_level:
    for i in range(100, 220, 5):
        ret, thr = cv2.threshold(img1, i, 255, 0)
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        cv2.imshow(str(i), img)
        file_name = 'img/output/'+str(i)+' .jpg'
        cv2.imwrite(file_name, img )
        impDef.close_window()

"""
    ret, thr = cv2.threshold(img1, 127, 255, 0)
   # _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
        ret, thr = cv2.threshold(imgray, 127, 255, 0)
        _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        Gray 스케일로 변환시킨 imgray를 thresholding 하여 그 값을 thresh로 합니다.
        이를 cv2.findContours() 함수에 넘겨 contour를 찾습니다
     

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    
          cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            cv2.drawContours() 함수는 우리가 찾은 contour를 실제로 그리는 함수이며, 5개의 인자를 가지고 있네요.
                img: contour를 나타낼 대상 이미지
                contours: img에 그릴 contour. 
                                   이 값은 cv2.findContours() 함수의 2번째 리턴 값으로 리스트형 자료임. 
                                   i번째 contour의 첫 번째 픽셀 좌표는 contours[i][0]과 같이 접근 가능
                -1: img에 실제로 그릴 contour 인덱스 파라미터. 이 값이 음수이면 모든 contour를 그림
                (0, 255, 0): contour 선의 BGR 색상값. 여기서는 Green으로 지정했음
                1: contour 선의 두께
    """
    #cv2.imshow('thresh', thr)
    #cv2.imshow('contour', img)
    #impDef.close_window()

contour()

"""
그러면 마지막으로 앞에서 언급했던 cv2.CHAIN_APPROX_SIMPLE과 
cv2.CHAIN_APPROX_NONE의 차이점을 가볍게 설명해 보겠습니다.

만약, 어떤 이미지에서 직선의 contour를 찾는다고 할 때, 
이 직선상의 모든 점을 얻고 그릴 필요는 없을 겁니다. 
직선의 양 끝점 2개만 있으면 되는 것이죠. 

앞에서 언급했듯이 cv2.CHAIN_APPROX_SIMPLE은 contour의 수평, 수직, 대각선 방향의 
직선상에 놓인 점들은 모두 버리고, 끝 점들만 취한다고 했습니다. 

cv2.CHAIN_APPROX_NONE은 contour의 모든 점을 취하는 것이고요.
이 두 방식의 차이는 컴퓨터 메모리를 얼마나 잡아 먹고 쓰느냐와 관련이 있고, 결국 성능 문제로 귀결될 겁니다.

이 두 방식이 가장 두드러지게 차이 나는 것을 보기 위해 검정색 바탕에 흰색 직사각형이 똑바로 놓여 
있는 이미지를 활용하여 contour를 그려보면 각각 아래와 같습니다.

참고로 아래 결과물은 위 두 가지 근사 방법에 따라 얻어진 contour의 모든 점에 원을 그린 것인데, 
이유는 그냥 contour를 획득해서 선으로 그리게 되면 cv2.CHAIN_APPROX_SIMPLE일 경우, 
아무것도 그려지지 않습니다. 왜냐하면 꼭지점 4개만 좌표로 가지고 있어서, 선을 그릴 수가 없죠.
"""


