"""
광학흐름

47. Lucas-Kanade
    cv2.calcOpticalFlowPyrLK(이전 프레임, 이전 검출 포인트들, 다음 프레임)

    Optical Flow응용
        1. 움직임을 통한 구조 분석
        2. 비디오 압축
        3. Video Stabilzation : 영상이 흔들렸거나 블러가 된 경우 깨끗한 영상으로 처리하는 기술

    Optical Flow의 가정
        1. 객체의 픽셀 intensity는 연속된 프레임 속에서 변하지 않는다.
        2. 이웃한 픽셀들 역시 비슷한 움직임을 보인다.

    Optical Flow 구현 로직
        1. cv2.goodFeaturesToTrack() : 비디오 이미지에서 추적할 포인트를 결정하기 위해
        2. 비디오에서 첫번째 프레임을 취하고 Shi-Tomasi 코너 검출을 수행
        3. 코너 검출을 통한 점에 대해 Lucas-Kanade Optical Flow를 이용해 반복적으로 점들을 추적

    아래 Sample 코드는 계산된 next 키포인트들이 정확한지 확인하지 않기 때문에 비디오 재생이 진행될 수록 추적할 특성 포인트들이 화면에서 사라질 수 있음.
    제대로 된 Optcal Flow를 구현하려면 특성 포인트들이 하나의 프레임이 아니라 특정 구간(보통 5프레임)마다 새롭게 검출해야 함.


48. Dense Optical Folw
    Lucas-Kanade 방법을 이용한 Optical Flow는 Shi-Tomasi 코너 검출을 활용하는 것으로 특성 포인트들이 조밀하지 못한 상태에서 계산된 것임
    OpenCV는 보다 조밀한 Optical Flow를 계산해주는 알고리즘을 제공하는데, 영상 프레임의 모든 포인트들에 대해서 Optical Flow를 계산한다.
    Lucas-Kanade의 방법에 비해 처리속도가 느린 단점이 있다.

    Sample Code
        영상속의 움직임을 움직임의 크기아 방향, 2개의 채널로 나타냄.
        움직임의 방향은 방향에 해당하는 Hue(색상)의 값으로 이미지를 나타내고 속도도는 Vlue(진하기)로 나타낸다.
        영상속에서 오른쪽으로 움직이는 물체는 빨간계통, 왼쪽으로 움직이는 물체는 녹색계통으로 나타난다.
        또한 물체가 빠르게 움직이면 진한 색상으로 나타난다.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners = 200, qualityLevel = 0.01, minDistance = 7, blockSize = 7)
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = termination)

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.blackscreen = False
        self.width = int(self.cam.get(3))
        self.height = int(self.cam.get(4))

    def run(self):
        while True:
            ret , frame = self.cam.read()
            if not ret :
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if self.blackscreen:
                vis = np.zeros((self.height, self.width, 3), np.uint8)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1,1,2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1,2).max(-1)
                good = d < 1
                new_tracks = []

                for tr, (x,y), good_flag in zip(self.tracks, p1.reshape(-1,2), good):
                    if not good_flag:
                        continue

                    tr.append((x,y))
                    if len(tr) > self.track_len:
                        del tr[0]

                    new_tracks.append(tr)
                    cv2.circle(vis, (x,y), 2, (0,255,0), -1)

                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x,y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x,y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x,y in np.float32(p).reshape(-1,2):
                        self.tracks.append([(x,y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            cv2.imshow('frame', vis)

            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break

            if k == ord('b'):
                self.blackscreen = not self.blackscreen

        self.cam.release()


def denseOptFlow():

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255

    while True:
        ret, frame = cap.read()
        next = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next, 0.0, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame', rgb)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        #end of if k == 27

        prev = next
    cap.release()
    # cap 객체 해제
    #end of while True
# End of def denseOptFlow()

def draw_flow(img, flow, step=16, black = False):
    global width, height

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x,y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if black:
        vis = np.zeros((height, width, 3))
    #End of if black

    cv2.polylines(vis, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    # End of for (x1, y1), (x2, y2) in lines

    return vis
# End of draw_flow()


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr
# End of draw_hsv(flow):

def warp_flow(img,flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[...,0] += np.arange(w)
    flow[...,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)

    return res
# End of warp_flow(img,flow):


def advanced_optflow():
   global width, height

   cap = cv2.VideoCapture(0)
   ret, prev = cap.read()
   prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
   show_hsv = False
   show_glitch = False
   cur_glitch = prev.copy()
   blackscreen = False
   width = int(cap.get(3))
   height = int(cap.get(4))

   while True:
       ret, frame = cap.read()
       if not ret:
           print('Not Work Camera!')
           break
        # End of if not ret

       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.0, 0.5, 3, 15, 3, 5, 1.1, 0)
       prevgray = gray

       frame2 = draw_flow(gray, flow, black = blackscreen)
       cv2.imshow('Grid Optical Flow', frame2)

       if show_hsv:
           vis_hsv = draw_hsv(flow)
           cv2.imshow('flow HSV', vis_hsv)
        # End of if show_hsv

       if show_glitch:
           cur_glitch = warp_flow(cur_glitch, flow)
           cv2.imshow('glitch', cur_glitch)
        # End of show_glitch

       ch = cv2.waitKey(60) & 0xFF
       if ch == 27:
           break

       if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV Flow Visualization is', ['Off', 'On'][show_hsv])
        # End of if ch == ord('1'):

       if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = frame.copy()
            print('glitch is', ['Off', 'On'][show_glitch])
        # End of if ch == ord('2'):

       if ch == ord('b'):
            blackscreen = not blackscreen
        # End of if ch == ord('b'):
    # End of while True:

   cap.release( )


# End of advanced_optflow():

#video_src = 0
#App(video_src).run()

#denseOptFlow()

advanced_optflow()
cv2.destroyAllWindows()









