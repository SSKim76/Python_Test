"""
Image Matching

42. OpenCV Labs의 ORB
    ORB(Oriented FAST and Rotated BRIEF)는 특허권이 있어 자유롭게 사용할 수 없는 SIFF와 SURF를 대체할 수 있도록
    OpenCV Labs가 개발한 이미지 특성 검출 알고리즘
    ORB는 키포인트 검출을 위해 FAST를, 알고리즘 성능 형상을 위해 많은 수정을 가한 BRIEF 디스크립터를 혼합 적용한 알고리즘

    ORB
        1. 키포인트를 찾기위핼 FAST를 사용
        2. 키포인트들 중 최상위 N개를 추출하기 위해 Harris 코너 검출방법 적용.
        3. 크기 불변 이미지 특성을 추출하기 위해 다양한 스케일의 피라미드를 적용
        3. 회전불변 특성을 추출하기 위해 조정된 BRIEF 디스크립터를 적용

43. ORB, SIFT를 이용한 이미지 특성 매칭
    두 이미지의 특성을 비교하는 방법
        1. 전수조사방법(BF매칭)
            A 이미지에서 하느의 특성 디스크립터를 취하고, 이 디스크립터를 B이미지의 모든 특성 디스크립터와 거리 계산 방법으로 하나하나 비교
            해서 나온 결과장 가장 비슷한 값을 리턴하는 방식으로 A 이미지의 모든 특성 디스크립터에 대해 계산한다.

            cv2.BFMatcher()
                normalType과 corssCheck 두개의 인자를 취할 수 있다.

                normalType -> 거리계산 방법 지정
                    - 보통 SIFT 또는 SURF인 경우 cv2.NORM_L2를 사용
                    - ORB, BRIEF인 경우 cv2.NORM_HAMMING을 사용하는 것이 좋다.

                crossCheck -> True 또는 False값이 지정되며, 디폴트는 False
                    - True : 두 이미지를 서로 BF 매칭하여 가장 일치하는 부분만을 리턴
                        이미지 A에서 i 번째 디스크립터를 취하여 이미지 B의 모든 디스크립터와 비교해 가장 일치하는 것이 j번째 일경우
                        거꾸로 이미지 B의 j번째 디스크립터와 A의 모든 디스크립터와 비교해 가장일치하는 것이 i번째가 되면 그 값을 리턴

44. FLANN 기반 이미지 특성 매칭
    FLANN(Fast Library for Approximate Nearest Neighbors)
        큰이미지에서 특성들을 매칭할 때 성능을 위해 최적화된 라이브러리 모음
        FLANN은 큰 이미지에 대해 BF매칭보다 빠르게 동작
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


def ORB(imgNo):
    img = cv2.imread(impDef.select_img(imgNo))
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = None

    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)

    img2 = cv2.drawKeypoints(img, kp, img2, (0,0,255), flags = 0)
    # img의 keypoint들을 img2에 표시

    cv2.imshow('ORB', img2)
    impDef.close_window()


def featureMatching(imgNo1, imgNo2):
    img1 = cv2.imread(impDef.select_img(imgNo1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(impDef.select_img(imgNo2), cv2.IMREAD_GRAYSCALE)
    res = None

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key =lambda x:x.distance)
    # metches의 요소들을 x.distance의 값으로 정렬. 즉 두 이미지의 특성 포인트들을 가장 일치하는 순서대로 정렬
    #res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], res, singlePointColor = (0,255,0), matchColor=(255,0,0), flags = 0)
    res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], res, flags = 0)
    # matches에서 처음 10개만 화면에 출력
    # flags = 0 : 매칭여부와 관계없이 두 이미지에서 찾은 특성 포인트 들을 모두 화면에 표시, 일치되는 포인트만 표시하려면 flags = 2
    # singlePointColor = (0,255,0), matchColor=(255,0,0) : 검출한 특성포인트는 초록색으로 일치하는 특성포인트는 파란색으로

    cv2.imshow('Feature Matching', res)
    return res


def SIFT_featureMatching(imgNo1, imgNo2):
    img1 = cv2.imread(impDef.select_img(imgNo1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(impDef.select_img(imgNo2), cv2.IMREAD_GRAYSCALE)
    res = None

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)
    res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], res, flags = 0)

    cv2.imshow('Feature Matching', res)
    return res


def FLANN(imgNo1, imgNo2, factor):
    img1 = cv2.imread(impDef.select_img(imgNo1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(impDef.select_img(imgNo2), cv2.IMREAD_GRAYSCALE)
    res = None

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    imdex_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # SIFT와 SURF를 활용하는 경우 indexParams는 예제코드와 같이 사전 자료를 생성
    # ORB를 활용하는 경우, 아래와 같은 방법으로 indexParams를 위한 사전 자료를 구성한다
    # index_params = dict(algorithm = FLANN_INDEX_LSH, table_number=6, keysize=12, multi_probe_level=1)
    # 이미지에 따라 위 설정갑들에 대해 ORB의 FLANN기반 매칭이 제대로 동작하지 않을 수 있다. 이경우 각각의 값들을 적절하게 조절하여야 한다.
    search_params = dict(checks = 50)
    # 특성매칭을 위한 반복회수. 값이 커지면 정확한 결과가 나오지만 속도는 느려진다.


    flann = cv2.FlannBasedMatcher(imdex_params, search_params)
    # FLANN 기반 매칭 객체를 생성
    matches = flann.knnMatch(des1, des2, k=2)
    # KNN(K-Nearesst Neighbor) 매칭 수행
    # K=2 : 2번째로 가까운 매칭 결과까지 리턴. Matches는 1순위 매칭결과, 2순위 매칭결과가 멤버인 리스트가 됨
    # KNN 매칭을 하는 이유는 리턴한 결과를 사용자가 선택하여 다룰 수 있기 때문

    good = []
    for m, n in matches:
        if m.distance < factor * n.distance:
            good.append(m)
    # matches의 각 멤버에서 1순위 매칭결과가 2순위 매칭결과의 factor로 주어진 비율보다 더 가까운 값만 취한다.
    # 즉 factor가 0.7 이므로 1순위 매칭결과가 2순위 매칭결과의 0.7배 보다 더 가까운 값만 취함.
    res = cv2.drawMatches(img1, kp1, img2, kp2, good, res, flags = 2)

    cv2.imshow('Feature Matching', res)




#ORB(42)
#featureMatching(42, 43)
#SIFT_featureMatching(42, 43)

sampleImg1 = featureMatching(44, 45)
#impDef.close_window( )
sampleImg2 = SIFT_featureMatching(44, 45)
#impDef.close_window( )

#rtnimg = cv2.vconcat([sampleImg1, sampleImg2])
rtnimg = np.vstack((sampleImg1, sampleImg2))

cv2.imshow('ORB & SIFT Matching Result', rtnimg)
impDef.close_window( )

FLANN(44, 45, 0.67)
impDef.close_window( )








