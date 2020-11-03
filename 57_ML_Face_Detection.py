"""
57강. 머신러닝 기초4 - 얼굴검출(Face Detection)

    Haar Feature 기반 cascade classifier(다단계 분류)를 이용한 객체검출 사용
    다수의 객체 이미지(Postive 이미지)와 객체가 아닌 이미지(negative 이미지)를 cascade함수로 트레이닝 시켜 객체를 검출하는
    머신러닝 기반의 접근 방식

    OpenCV는 Haar-cascade 트레이너와 검출기를 모두 제공


"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef

font = cv2.FONT_HERSHEY_SIMPLEX

def faceDetect():

    eye_detect = False
    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
    # 얼굴 검출을 위한 Haar-Cascade 트레이닝 데이터를 읽어 CascadeClassifier 객체를 생성
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # 눈 검출을 위한 Haar-Cascade 트레이닝 데이터를 읽어 CascadeClassifier 객체를 생성

    info = ''

    try:
        cap = cv2.VideoCapture(0)
    except:
        print('카메라 로딩 실패!!')


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if eye_detect:
            info = 'Eye Detection ON'
        else:
            info = 'Eye Detection OFF'
        # End of if eye_detect

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # CascadeClassifier의 detectMultiScale 함수에 gray scale 이미지를 입력하여 얼굴을 검출
        # 얼굴이 검출되면 위치를 리스트로 반환
        # 위치는 (x, y, w, h)와 같은 튜플이며 (x, y)는 검출 좌상단 위치, (w, h)는 가로, 세로 크기
        # ScaleFactor = 1.3
        # minNeighbor = 5

        cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

        for(x, y, w, h) in faces:
          cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
          cv2.putText(frame, 'Detected Face', (x-5, y-5), font, 0.5, (255, 255, 0), 2)

          if eye_detect:
              roi_gray = gray[y:y+h, x:x+w]
              roi_color = frame[y:y+h, x:x+w]
              eyes = eye_cascade.detectMultiScale(roi_gray)
              for(ex, ey, ew, eh) in eyes:
                  cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
              # End of for(ex, ey, ew, eh) in eyes
          # End of if eye_detect
        # End of for(x, y, w, h) in faces

        cv2.imshow('frame', frame)
        k = cv2.waitKey(30)
        if k == ord('i'):
            eye_detect = not eye_detect
        if k == 27:
            break
    # End of while
    cap.release()
    cv2.destroyAllWindows()
# End of FaceDetect()

faceDetect()




