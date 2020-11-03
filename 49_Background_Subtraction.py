"""
49강. 영상에서 배경제거하기

    배경제거는 많은 종류의 비전 기반 어플리케이션에서 활용되는 주된 전처리 프로세싱 과정
    CCTV 와 같은 고정된 카메라를 이용해서 출입하는 방문객 수를 계산하는 프로그램이나, 교통량 조사를 위한 프로그램에서 활용
    보통 고정된 카메라에서 촬영하는 영상의 경우, 배경이 되는 부분은 움직이지 않는 정지영상이고, 출입하는 사람이나 자동차등은 움직이는 객체
    따라서 기술적으로 배경제거는 정지된 부분에서 움직이는 부분만 추출하면 가능함.
    움직이는 객체의 그림자가 포함되어 있으면 배경을 추출하는일이 복잡해짐.

    OpenCV에서 제공하는 알고리즘
        1. BackgroundSubtractorMOG
            가우시안 믹스쳐 기반 배경/전경 분할 알고리즘
            K값이 3 또는 5인 가우시안 분포 믹스쳐를 배경 픽셀에 적용함으로써 배경제거를 수행
            믹스쳐에 대한 가중시는 영상에서 배경 제거를 위한 특성 픽셀이 동일한 장소에 머물고 있는 시간 비율을 나타냄.

        2. BackgroundSubtractorMOG2
            가우시안 믹스쳐 기반 배경/전경 분할 알고리즘
            각 픽셀에 적절한 가우시안 분포값을 선택
            조명상태의 변화로 장면이 변해도 제대로 배경을 제거함
            detectShadow = True ; 그림자 검출설정(Default = True), 제외하려면 False
            detectShadow = True ; 그림자는 회색으로 표시
            그림자 검출을 설정하면 처리속도는 약간 느려짐짐

        3 BackgroundSubtractorGMG(다양상 상황에서 가장 훌륭하게 배경제거를 해준다고 알려져 있음)
            통계적 배경 이미지 제거와 픽셀 단위 베이지안 분할을 결합한 알고리즘
            GMG 알고리즘은 최초 몇 프레임(보통 120프레임)을 배경 모델링을 위해 사용
            배경이 아닌 전경이나 움직으는 객체를 추출하기 위해 베이지안 추론을 이용
            노이즈를 제거하기 위해 opening 기법을 적용하는 것이 좋다고 알려져 있음


"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


def backSubtractionMOG():
    cap = cv2.VideoCapture(0)
    #cap.set(3, 480)
    #cap.set(4, 320)

    mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    mog2 = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        fgmask = mog.apply(frame)
        fgmask2 = mog2.apply(frame)

        cv2.imshow('MOG', fgmask)
        cv2.imshow('MOG2', fgmask2)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    # End of while True:
# End of backSubtractionMOG():



def backSubtractionGMG():
    cap = cv2.VideoCapture(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    while True:
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('GMG', fgmask)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # End of while Ture
    cap.release()
    cv2.destroyAllWindows()
# End of backSubtractionGMG()




backSubtractionGMG()
#backSubtractionMOG()





