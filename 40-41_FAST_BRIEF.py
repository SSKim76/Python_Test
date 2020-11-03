"""
40. 코너 검출을 위한 FAST 알고리즘

    cv2.FastFeatureDetector_create()
    FAST 알고리즘을 이용한 특징 검출 원리
        1. 이미지에서 픽셀 p를 선택, 선택한 p의 밝기를 Ip라고 함
        2. 적당한 threshold값 t를 취함
        3. 픽셀 p주변의 16개 픽셀로 이루어진 원을 생각함
        4. 16개 픽셀로 이루어진 원 내부에 IP + t 보다 밝은 픽셀인 n개 연족으로 존재하거나,
            IP - t 보다 n개 연속으로 존재하면 픽셀 p는 코너로 판단
        5. 고속 테스트를 통해 코너가 아닌 픽셀은 제외함.
            고속 테스트는 상, 하, 좌, 우 4개의 픽셀을 테스트 함. p가 코너라면 4개의 픽셀 중 적어도 3개는 Ip + t 보다
            밝거나, Ip - t보다 어두워야 함.
            5번 과정은 성능이 매우 좋지만 다음과 같은 약점이 존재함.
                1. n < 12 인 경우 테스트 하려는 픽셀 후보들을 많이 버리지 못함.
                2. 이미지 특징 검출 효율성이 코너 모양의 분포나 질의하는 순서에 의존적이기 때문에 픽셀 선택이 최적이 아님
                3. 코속 테스트의 결과들은 버려짐
                4. 코너 주위로 다중 특징들이 검출되기도 함함
            처음 3가지는 머신 러닝을 통해 극복하게 되면, 마지막 약점은 non-maximal suppression을 이용하여 극복한다.

    코너검출을 위한 머신러닝
        1. 훈련을 위한 이미지 세트를 선택함.
        2. 특성을 찾기 위해 모든 이미지에 FAST알고리즘을 실행함
        3. 검출된 모든 픽셀에 대해, 픽셀 주위로 16개 픽셀을 벡터로 저장함. 이를 벡터 P라고 함
        4. 16개의 각픽셀(x)은 아래 3가지 상태중 하나가 됨(Darker, Similar, Brighter)
        5. Darker :  Ip -> Ix <= Ip -t
            Similar : Ip -t < Ip -> Ix < Ip + t
            Brighter : Ip + t <= Ip -> Ix
        6. 이 상태에 따라 벡터 P를 Pd, Ps, Pb 세가지로 구분함
        7. 새로운 boolean 변수 Kp를 정의. Kp는 p가 코너이면 True, 아니면 False
        8. Pd, Ps, Pb에 Kp변수를 이용해 의사결정 트리 기반 분류 아고리즘인 ID3을 적용함.
            ID3은 머신러닝이나 자연어 처리에 전형적으로 사용되는 알고리즘 임.
            Kp의 엔트로피를 측정함으로서 후보 픽셀이 코너인지 아닌지에 관해 많은 정보를 제공하는 픽셀x를 선택함.
        9. Kp의 엔트로피가 0이 될 때까지의 모든 Pd, Ps, Pb에 리커시브 하게 적용함
        10. 다른 이미지들에 대해 코너를 빠르게 검출할 수 있는 의사결정 트리가 생성됨.

    Non-maximal Supporession
        1. 검출된 모든 특성 포인트들에 대해 스코어 함수 V를 계산함.
            V는 p와 이를 둘러싼 16개의 픽셀 값의 차의 절대값을 모두 더한 값
        2. 근접한 2개의 키포인트에 대해 V를 계산함.
        3. V값이 작은 쪽을 버림림

    FAST는 다른 코너 검출 알고리즘에 비해 몇배 정도 빠름. 하지만 이미지에 노이즈가 많을 경우 제대로 된 결과가 나오지 않고,
    threshold 값에 의존적 임.
    cv2.FastFeatureDetector_create()



41. 이미지 디스크립터 고속계산 - BRIEF

    디스크러터 1개당 SIFT는 512Byte, SURF는 256바이트의 메모리를 필요로 함. 수천개 이상의 특성이 있는 이미지의 경우 많은 메모리를 필요로 함.
    실제로 이미지들을 매칠할 때 모든 특성들이 필요한 것은 아님.
    선형분석법(LDA), 주성분분석법(PCA)과 같은 기법을 활용하여 특성들을 축약하여 계산할 수 있음.
    LSH(Locality Sensitive Hashing)를 이용하는 해싱 기법은 부동소수점으로 되어있는 SIFT 디스크립터들을 바이너리 문자열로 변환하는데 사용하기도 함.
    부동소수점 형태의 데이터를 바이너리 문자열로 변환하게 되면 고속처리가 가능 함. 이는 해밍거리를 찾기 위해 비트간 XOR 연산만으로 계산 가능하기 때문.
    해밍거리란 문자열에서 발생하는 오류를 측정하는 방법중 하나임.
    이런 방법을 활용하더라도 먼저 특성 디스크립터를 찾아서 해싱을 적용해야 하므로 메모리 문제를 해결해 주는 것은 아님.

    BRIEF는 특성 디스크립터 검출 필요 없이 바이너리 문자열을 구하는 빠른 방법.
    BRIEF로 이미지 디스크립터에 대한 바이너리 문자열을 구하면 해밍거리를 이용해 디스크립터들을 매치
    BRIEF는 하나의 특성 디스크립터이며, 이미지 특성들을 검출하기 위한 다른 방법은 제공하지 않음.
    따라서 이미지 특성 검출을 위해서는 SIFT나 SURF와 같은 이미지 특성 검출방법과 같이 사용해야 함.
    아래 예제는 SURF보다 좀더 나은 성능을 보이는 CenSurE 고속검출기를 함께 사용

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


def FAST(ImgNo):
    img = cv2.imread(impDef.select_img(ImgNo))
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2, img3 = None, None

    fast = cv2.FastFeatureDetector_create(30)
    # 인자값 30은 threshold 값, 값이 커지면 검출하는 키포인트의 수는 줄어듬.

    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, img2, (255,0,0))
    # Non-maximal Suppression -> Ture
    cv2.imshow('Fast1', img2)

    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    img3 = cv2.drawKeypoints(img, kp, img3, (0,0,255))
    # Non-maximal Suppression -> False
    cv2.imshow('FAST2', img3)

    impDef.close_window()



def BRIEF(imgNo):
    img = cv2.imread(impDef.select_img(imgNo))
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = None

    # STAR 검출기 초기화
    star = cv2.xfeatures2d.StarDetector_create()
    #OpenCV에서는 CenSurE 특성검출기를 STAR 검출기로 부름.

    # BRIEF 추출기 초기화
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # STAR로 키포인트를 검출하고 BRIEF로 디스크립터 계산
    kp1 = star.detect(img, None)
    kp2, des = brief.compute(img, kp1)

    img2 = cv2.drawKeypoints(img, kp1, img2, (255,0,0))
    cv2.imshow('BRIEF', img2)

    impDef.close_window()



#FAST(37)
BRIEF(37)







