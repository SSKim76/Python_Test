"""
37강. 코너검출(Detection Corner)
  이미지가 회전하더라도 코너를 검출 할 수 있지만, 이미지의 크기가 커 지면 제대로 검출하지 못함
  이미지가 작을때는 코너로 인식하지만, 이미지가 커저 평탄하게 되기 때문.

  Harris Corner Detection
    cv2.cornerHarris(imgray, Tile, Sobel, Value)
    imgGray : 검출대상 그레이스케일 이미지, Float32 Type
    Tile : 코너 검출을 위해 고려할 이웃 픽셀의 범위
    Sobel : Sobel 미분에 사용된 인자 값
    Value : Harris 코너 검출 수학식 R에서 K 값


  Shi-Tomasi Corner Detection
    cv2.goodFeatureToTrack(grayImg, qty, thr, min)
    grayImg : 검출대상 이미지, 그레이 스케일
    qty : 검출할 코너의 갯수
    thr : 코너 검출 품질 - 코너로 판단할 문턱 값
    min : 검출할 코너 사이의 최소거리. 이 거리 이내에 있으면 무시함.



38강. 크기불변 이미지 특성 검출기법 SIFT

  SIFT
    이미지의 크기가 달라지더라도 이미지의 특징적인 부분을 검출하는 기법
    이미지에서 스케일 불변인 키포인트를 추출하고, 추출한 키포인터들의 descriptor를 계산
    SIFT는 특허에 등록된 알고리즘 임. 따라서 OpenCV무료버전에는 SIFT의 모든 알고리즘이 탑제되어 있지 않음
    OpenCV3.0이상 버전에서는 OpenCV contrib 버전을 반드시 설치해야 함.

  절차
    1. Scale-Space Extrema Detection(스케일-공간 극값 검출)
      가우시안 필터 후, 라플라시안 필터(Laplacian of Gaussian : LOG)를 적요하면 이미지에서 다양한 크기의 방울모양 의 이미지를 검출
      하지만 LOG는 다소 시간이 소요됨. 이에 SIFT알고리즘에서는 하나의 이미지에 서로 다른 필터를 적용한
      가우시안 피라미드 이미지의 차(Difference of Gaussian : DOG)를 이용
      DOG를 찾으면 이미지에서 스케일-공간 좌표상 극값을 찾음. 만약 극값이 있으면 이를 잠재적 키포인트(Potential Keypoint)라고 한다.

    2. Keypoint Localization(키포인트의 지역화)
      이미지에서 잠재적 키포인트들의 위치를 모두 찾았으면 보다 정확한 결과를 위해 잠재적 키포인트들의 정제과정을 거켜 키포인트들을 추출
      정제과정은 테일러 전개를 이용하여 수행

    3. Orientaltion Assignment(방향 할당하기)
      최종적으로 추출된 키포인트들에 방향성-불변이 되도록 방향을 할당.
      즉 이미지가 확대되거나 회전하더라도 추출된 키포인트들은 이미지의 특징을 보존하게 됨.

    4. Keypoint Descriptor(키포인트 디스크립터 계산)
      키포인트를 이용하여 키포인트 디스크립터를 계산
      이미지 히스트그램을 활용하여 표현. 이외 조명의 변화나 회전 등에도 키포인트득이 특징을 보존할수 있도록 몇가지 측정값을 추가

    5. Keypoint Matching(키포인트 매칭)
      두 개의 이미지에서 키포인트들을 매칭하여 동일한 이미지 추출이나 이미지 검색 등에 활용

  SIFT를 위해 OpenCV에서 제공하는 함수
    cv2.xfeatures2d.SIFT_create()객체 : SIFT의 키포인트, 디스크림터들을 계산하는 함수를 제공
    detect(grayimg) : grayimg에서 키포인트를 검출하여 리턴
    compute(keypoint) : keypoint에서 디스크립터를 계산한 후 키포인트와 디스크립터를 리턴
    detectAndCompute(grayimg) : grayimg에서 키포인트와 디스크립터를 한번에 계산하고 리턴
    cv2.drawKeypoints(grayimg, keypoints, outimg) : outimg에 grayimg의 keypoint들을 표시시




39강. SIFT의 성능 향상 버전 - SURF

  SURF(Speeded-Up Robust Features)
    SIFT는 이미지 특징을 찾아내는 훌륭한 알고리즘이지만 상대적으로 성능이 좋지 않음.
    SURF는 박스필터로 LoG를 근사하는 방법을 사용함
    SURF는 SIFT에 비해 약 3배 이상 속도가 빠름
    SURF는 블러이미지나 회전된 이미지의 경우 이미지 특징을 제대로 잡아내지만, 뷰 포인트가 바뀌거나 조명이 달라지면 제대로 검출하지 못함

    이미지 특징을 비교할 때 회전이 문제 되지 않을 경우
    예를 들면 파노라마 사진에서 비슷한 물체 찾기 등과 같은 경우
    키포인트 검출 시에 회전 요소를 제거하면 더 빠르게 결과를 보여줌
    회전 요소를 제거하려면 아래의 코드를 키포인트 검출 이전에 추가합니다.
    surf.setUpright(True)

    OpenCV가 제공하는 SURF 객체는 디스크립터를 64차원 또는 128차원 크기의 벡터로 설정하고 초기화 할 수 있음.
    SURF 객체의 디스크립터 크기를 알고자 할 때는 다음과 같이 한다.
    surf.descriptorSize() : 이 값이 64이면 이미지 매칭을 위해 128 차원으로 변경하여야 하는데 다음과 같이 수행

    surf.setExtended(True)
    kp, des = surf.detectAndCompute(img, None) : 이렇게 하면 디스크립터의 크기가 128로 변경됨.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


def detectCornerHarris(imgNo):
    img = cv2.imread(impDef.select_img(imgNo))
    img2 = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgGray = np.float32(imgGray)
    dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)
    # 검출된 코너부분을 확대하기 위해
    dst = cv2.dilate(dst, None)

    # 원본에 적적할 부분을 빨간색으로 표시
    # dst.max() 앞에 곱한 상수를 적절하게 조절하면 검출된 코너를 최적화 하여 나타 낼 수 있음
    img2[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('Harris', img2)
    impDef.close_window()




def shito(imgNo):
    img = cv2.imread(impDef.select_img(imgNo))
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(grayImg, 25, 0.01, 10)
    corners = np.int0(corners) # 정수형 값으로 전환

    for i in corners:
        x , y = i.ravel()
        cv2.circle(img, (x,y), 3, 255, -1)

    cv2.imshow('shito', img)
    impDef.close_window()


def SIFT(imgNo):
    img = cv2.imread(impDef.select_img(imgNo))
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2, img3 = None, None

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(grayImg, None)

    img2 = cv2.drawKeypoints(grayImg, kp, img2)
    img3 = cv2.drawKeypoints(grayImg, kp, img3, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('SIFT1', img2)
    impDef.close_window( )

    cv2.imshow('SIFT2', img3)
    impDef.close_window( )
    #img4 = cv2.hstack((img2, img3))
    img4 = cv2.hconcat([img2, img3])
    cv2.imshow('비교', img4)
    impDef.close_window()


def SURF(imgNo):
    img = cv2.imread(impDef.select_img(imgNo))
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2, img3 = None, None

    # 회전 요소를 제거하면 더 빠른 결과를 보여줌
    # 회전요소를 제거하려면 아래의 코드를 키포인트 검출 이전에 추가
    # surf.setUpright(True)

    surf = cv2.xfeatures2d.SURF_create()  # SURF 객체 생성
    surf.setUpright(True)  # 회전요소 제거, 회전이 문제가 되지 않는 경우 사용
    surf.setHessianThreshold(10000)
    # 인자값에 따라 SURF 객체가 검출하는 키포인트의 개수가 달라짐
    # 값이 작아지면 검출하는 키포인트의 개수가 많아지고, 값이 커지면 키포인트 개수는 적어짐.
    # 만약 두개의 이미지를 비교하거나 이미지에서 특정 사물을 추출하고자 할 때 적절한 인자값은 300 ~ 500 사이
    # 위 두줄은 surf = cv2.xfeatures2d.SURF_create(10000)과 동일

    kp, des = surf.detectAndCompute(img, None)
    img2 = cv2.drawKeypoints(grayImg, kp, img2, (255, 0, 0), 4)
    img3 = cv2.drawKeypoints(img, kp, img3, (255, 0, 0), 4)

    cv2.imshow('SURF', img2)
    cv2.imshow('SURF2', img3)
    impDef.close_window()


#detectCornerHarris(37)
#shito(37)
#SIFT(38)
SURF(38)