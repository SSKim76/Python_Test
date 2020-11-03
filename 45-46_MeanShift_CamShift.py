"""
비디오에서 객체 추적하기

45. MeanShift
    점들의 집합에서 초기에 잡은 원 영역을 C1, C1의 중심을 C1_o, C1의 무게중심을 C1_r이라고 하면,
    아마 99%이상 C1_o와 C1_r은 일치하지 않을것임.
    이때 C1_0를 C1_r과 일치되도록 C1을 이동 -> 다시 C1_o와 C1_r을 계산. 이런 과정을 반복하여
    C1_o와 C1_r이 최소차로 일치하게 하면 이 원 영역의 점들의 밀도는 가장 높게 됨

    실제 MeanShift를 적용하려면 히스토그램 배경투가가 된 이미지와 이 이미지에서 객체의 최초 위치를 지정하여 전달해야 함.
    객체가 움직이기 시작하면 히스토그램 배경투사된 이미지에 객체의 움직임이 반영됨.
    MeanShift는 객체 영역의 픽셀 밀도와 가장 일치하는 영역을 찾아주게 되어 최초 지정된 객체 영역을 추적할 수 있다.

    MeanShift 절차
        1. 타겟설정
        2. 설정된 타켓의 히스토그램을 구한다.
            Meanshift 계산을 위해 비디오의 각 프레임에서 타겟의 배경투사를 할 수 있도록
        3. 초기 타겟 영역의 위치를 지정
        4. 히스토그램은 색공간에서 Hue(색상)만 고려한다.
    단점
        추적하는 물체가 가까워지거나 멀어져도 일정한 영역 크기로 추적한다. 이로인해 객체 주적이 원활하게 이루어지지 않을 수도 있다.


46. CamShift
    MeanShift의 문제점을 수정한 알고리즘
    객체를 추적하기 위한 영역의 키기를 객체까지 거리나 객체의 형상에 따라 유연하게 바꿈

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


col, width, row, height = -1, -1, -1, -1
frame = None
frame2 = None
inputmode = False
rectangle = False
trackWindow = None
roi_hist  = None

# camShift 함수에서 'i'가 입력되었을때 마우스로 추적할 객체를 지정할 수 있도록 구현
def onMouse(event, x, y, flags, param):
    global col, width, row, height, frame, frame2, inputmode
    global rectangle, roi_hist, trackWindow

    #print('On Mouse')

    if inputmode:
        #print('input Mode')

        if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼이 클릭되었을 때 처리할 루틴
            rectangle = True
            col, row = x, y

        elif event == cv2.EVENT_MOUSEMOVE:  # 왼쪽 마우스 버튼을 누른채 움직일째 처리할 루틴
            if rectangle:
                frame = frame2.copy()
                cv2.rectangle(frame, (col, row), (x,y), (0,255,0), 2)
                cv2.imshow('frame', frame)

        elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼을 땠을때 처리할 루틴
            inputmode = False
            rectangle = False
            cv2.rectangle(frame, (col, row), (x,y), (0,255,0),2)
            # 마우스를 누르고 이동한 거리를 녹색으로 표시
            height, width = abs(row - y), abs(col-x)
            # 마우스를 누르고 움직인 이동거리 계산
            trackWindow = (col, row, width, height)
            roi = frame[row:row+height, col:col+width]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # ROI를 HSV 색공간으로 변경
            roi_hist = cv2.calcHist([roi], [0], None, [180], [0,180])
            # HSV 색공간으로 변경한 히스토그램을 계산
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            # 계산된 히스토그램을 노멀라이즈 함.
    return

def meanShift():

    global frame, frame2, inputmode, trackWindow, roi_hist

    try:
        cap = cv2.VideoCapture(0)
    except Exception as e:
        print(e)
        return

    #videoWidth = 320
    #videoHeight = 240
    #cap.set(3, videoWidth)  # 해상도 설정
    #cap.set(4, videoHeight)

    ret, frame = cap.read()
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse, param = (frame, frame2))

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # meanShift의 3번째 인자를 정의 : iteration회수를 10회, C1_o와 C1_r의 차가 1pt일 때까지 알고리즘을 구동하라는 의미
    while True:
        ret, frame = cap.read()
        if not ret :
            break

        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # roi_hist는 normalize ROI 히스토그램
            ret, trackWindow = cv2.meanShift(dst, trackWindow, termination)

            x, y, w, h = trackWindow
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            # 추적된 물체를 녹색 사각형으로 표시시
        cv2.imshow('frame', frame)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break

        if k == ord('i'):
            print('Select Area for MeanShift and Enter a Key')
            inputmode = True

            # 'i' 키를 눌렀을때 OnMouse에 구현된 로직이 활성화 됨
            frame2 = frame.copy()

            while inputmode:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
                # 현재화면을 키보드를 누를때 까지 일시 멈추게 함
    cap.release()
    cv2.destroyAllWindows()



def camShift():
    global frame, frame2, inputmode, trackWindow, roi_hist

    try:
        cap = cv2.VideoCapture(0)
        #cap.set(3, 480)
        #cap.set(4, 320)
    except Exception as e:
        print('카메라 구동 실패 ; ', e)
        #print(e)
        return

    ret, frame = cap.read( )
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse, param = (frame, frame2))

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # meanShift의 3번째 인자를 정의 : iteration회수를 10회, C1_o와 C1_r의 차가 1pt일 때까지 알고리즘을 구동하라는 의미
    while True:
        ret, frame = cap.read( )
        if not ret:
            break

        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # roi_hist는 normalize ROI 히스토그램


            # 여기부터 수정되는 듯
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            #x, y, w, h = trackWindow
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 추적된 물체를 녹색 사각형으로 표시시
        cv2.imshow('frame', frame)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break

        if k == ord('i'):
            print('추적할 영역을 지정한 후, 아무키나 누르세요')
            inputmode = True

            # 'i' 키를 눌렀을때 OnMouse에 구현된 로직이 활성화 됨
            frame2 = frame.copy( )

            while inputmode:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
                # 현재화면을 키보드를 누를때 까지 일시 멈추게 함
    cap.release( )
    cv2.destroyAllWindows( )

#meanShift()
camShift()










