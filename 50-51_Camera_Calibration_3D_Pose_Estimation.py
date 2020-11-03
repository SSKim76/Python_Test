"""
50강. 증강현실 기초1 - Camera Calibration
    이미지 왜곡의 종류
        1. Radial Distoration
            실제로는 직선인 모양이 곡선으로 왜곡되는 현상
        2. Tangential Distorion
            렌즈가 포착한 이미지가 상이 맻히는 이미지면과 평행하게 정렬되지 않아서 발생하는 왜곡

    증강현실이나 홀로그래픽과 같은 스테레오 비전(Streo Vision)을 다루는 어플리케이션에서는
    가장 먼저 이미지 왜곡현상을 제거하기 위해 캘리브레이션을 실행

51강. 증강현실 기초2 - 3D Pose Estimation
    Vision 영역에서의 poes란?
        3D Pose Estimatin이란 3차원 객체를 표현하고 있는 이차원 이미지에서 이 객체가 어떻게 변화되는가에 관한 문제로 귀결
        2차원 화면에서 3차원 객체를 표현하려면 객체를 표현하기 위한 기준점이 되는 좌표계가 있어야 한다.
        모니터 화면은 2차원이고, 눈앞에 고정되어있다. 만약 화면상의 가상의 좌표계가 움직인다면,
        이 좌표계에서 표현되는 물체도 동일한 POES로 움직인다.

    체스판 모양의 격자의 움직임에 따른 가상의 정육면체를 표시하는 PGM
        1. Camera Calibration 값을 저장한 파일에서 mtx, dist의 값을 얻는다.
        2. cv2.findChessBoardCorners()로 체스판의 격자점들의 정보를 얻는다.
        3. 2번의 절차가 성공적으로 수행되면 cv2.cornerSubPix()로 체스판 격자점에 대한 정확도를 높여준다.
        4. cv2.solvePnPRansac()을 이용해 객체의 회전 및 이동에 관한 정보, 즉 Pose를 계산한다.
        5. cv2.projectPoints()를 이용해 실제 공산상의 객체의 Pose정보를 2차원 화면상의 이미지의 Pose 정보로 변환한다.
        6. 화면상에 계산된 Pose정보를 표시한다.


"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef



# Camera Calibration
# 캘리브레이션을 위해 유효한 15장을 이미지를 찍고 이를 기준으로 카메라 왜곡 계수를 구하여 파일로 저장
def saveCamCalibraion():

    try:
        cap = cv2.VideoCapture(0)
        #cap.set(3, 480)
        #cap.set(4, 320)
    except Exception as e:
        print('카메라 구동 실패 ; ', e)
        #print(e)
        return

    ret, frame = cap.read( )
    cv2.imshow('Camera', frame)
    termination = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    objp = np.zeros((7*10, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:10].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray', gray)
        ret, corners = cv2.findChessboardCorners(gray, (7,10), None)
        print('ret = ', ret)
        #print('corners = ', corners)

        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), termination)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, (7,10), corners, ret)
            count = count+1
            print('[%d]'%count)
        # End of ret:
        cv2.imshow('img', frame)
        k = cv2.waitKey(0)
        if k == 27:
            break

        if count > 15:
            break
    # End of while True

    cv2.destroyAllWindows()
    ret, mtx, dits, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez('calib.npz', ret = ret, mtx = mtx, dist = dits, rvecs = rvecs, tvecs = tvecs)
    print('카메라 캘리브레이션 데이터를 저장했습니다.')
# End of saveCamCalibraion():

def  drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), -3)

    for i, j in zip(range(4), range(4,8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)
    # End of for i, j in zip(range(4), range(4,8)):
    cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)

    return img
# End of drawCube(img, corners, imgpts):


def poseEstimation():
    with np.load('calib.npz') as X:
        ret, mtx, dist, _, _ = [X[i] for i in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]

    termination = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((7*10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:10].T.reshape(-1,2)
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3], [0,3,-3], [3,3,-3], [3,0,-3]])

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(0)


    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 10), None)
        print('ret = ', ret)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), termination)
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            frame = drawCube(frame, corners, imgpts)
        # End of ret == True:
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cap.release()
    # End of  While True:

# End of def poseEstimation():



#saveCamCalibraion()
poseEstimation()

impDef.close_window()
cap.release()