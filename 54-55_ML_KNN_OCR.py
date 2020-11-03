"""
54강. 머신러닝 기초1 - kNN 이해하기

    머신러닝은 컴퓨터나 기계로 주어진 데이터를 바탕으로 새로운 질문에 대해 혜측하는 것을 목적으로 함

    1. 머신러닝의 방법
        1-1. 지도학습(Supervised Learning)
            학습을 위해 주어진 데이터를 어떤 조건을 만족하는 경우, 그에 대한 정답을 미리 제시해 둠.
            즉 어떤 규칙에 의한 룰 기반으로 학습을 시킴
        1-2. 비지도학습(NonSupervised Learning)
            학습을 위해 주어진 데이터에 정답을 제시하지 않고 컴퓨터 스스로 알아서 정답을 찾아내는 방법
            예) 승용차, 화물차, 버스 등에 대한 모양을 학습시킨 후, 다양한 차종이 섞여 있는 이미지에서 승용차, 버스 등을 구분하는 것 등

    2. kNN(k-Nearest Neighbours)
        지도학습에 활용되는 가장 단순한 종류의 알고리즘



55강. 머신러닝 기초2 - kNN을 이용하여 손글씨 인식하기
    OCR(Optical Character Recognition) 은 인쇄된 글자나 손으로 쓴 글씨를 인식할 수 있는 코드로 변환하는 기술

    1. Sample Image
        1-1. 총 5,000자(0~9까지 각각 500개)
        1-2. 각 숫자의 크기 : 20 x 20 Pixel
        1-3. 각 숫자는 가로 : 100 , 세로 5로 구성
        1-4. 총 해상도 2,000 x 1,000 Pixel

    2. 초기데이터 학습 로직
        2-1. sample Image 에서 20x20 크기의 셀로 구분하여 총 5,000개를 구성
        2-2. 20x20 크기의 셀에 있는 픽셀값을 1차원으로 배열(측 400개의 값)
        2-3. 2-2에서 1차원으로 구성한 400개의 값은 0에 해당하는 것이 500개, 1에 해당하는 것이 500개 9까지 총 5,000개
                따라서 배열로 나타내면(5000, 400)의 크기가 됨. 편의상 traindata
        2-4. traindata[:500,:]은 0에 해당하므로 0으로 표시, traindata[500:1000, :] 은 1에 해당하므로 1로 표시, 9까지 반복
                이는 컴퓨터로 하여금 이런 손글씨는 1이야, 2야 하는 식으로 가르치기 위한 것
                traindata에 이런 표시를 할 수 없으므로, (5000,1)가진 배열에다 순서대로 0을 500개, 1을 500개 ~ 9까지 구성하면 됨
        2-5. 학습한 Data를 재 사용하기 위해 4까지 학습한 내용을 파일로 저장

    3. 손글씨 숫자인식 로직
        3-1. 학습한 내용이 저장된 파일을 읽는다.
        3-2. 손글씨로 적은 숫자 이미지를 인자로 받고, 이를 20x20 픽셀 크기로 변환
        3-3. kNN을 이용해 학습한 내용을 바탕으로 20x20 픽셀 크기로 변환한 이미지를 인식

    4. 재학습 로직
        4-1. 실제 손글씨 숫자와 인식한 숫자가 다르면 이 손글씨에 대해 재대로 학습을 시킨다.
        4-2. 학습시킨 결과를 초기 학습 데이터에 추가한다.
        4-3. 저장한 학습 데이터를 갱신하여 저장한다.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef



def makeTraindata():
    traindata = np.random.randint(0, 100, (25,2)).astype(np.float32)
    # 0 ~ 100 범위에서 2차원 좌표로 된 25개의 멤버를 랜덤하게 생성
    resp = np.random.randint(0,2, (25,1)).astype(np.float32)


    return traindata, resp
# End of makeTraindata()

def knn():
    traindata, resp = makeTraindata()

    red = traindata[resp.ravel() == 0]
    blue = traindata[resp.ravel() == 1]
    # 각 멤버들을 랜덤하게 클래스0(빨간 삼각형), 클래스1(파란 사각형)으로 구분

    plt.scatter(red[:,0], red[:,1], 80, 'r', '^')       # 빨간 삼각형으로 표시
    plt.scatter(blue[:,0], blue[:,1], 80, 'b', 's')     # 파란 사각형으로 표시

    newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
    # 새로운 멤버변수 생성
    plt.scatter(newcomer[:,0], newcomer[:,1], 80, 'g', 'o') # 녹색 원으로 표시
    plt.show()

    knn = cv2.ml.KNearest_create()
    knn.train(traindata, cv2.ml.ROW_SAMPLE, resp)
    ret, results, neighbours, dist = knn.findNearest(newcomer, 5)
    # kNN으로 k=3일때 이 새로운 멤버가 어느 클래스에 속하는지 알아보는 것

    print(results, neighbours)

    return
# End of knn()


def resize20(digitImg):
    # 손글씨 숫자 이미지 파일을 인자로 받아 20x20 픽셀로 변환 후, 인식을 위해(1,400)크기의 numpy 배열로 리턴
    img = cv2.imread(digitImg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = cv2.resize(gray, (20,20), fx = 1, fy = 1, interpolation = cv2.INTER_AREA)

    ret, thr = cv2.threshold(ret, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('ret', thr)

    return thr.reshape(-1, 400).astype(np.float32)
# End of resize20(digitImg)


def learningDigit():
    # 초기 데이터 학습을 위한 함수
    # 학습한 내용은 digits_for_ocr.npz로 저장
    img = cv2.imread('img/OCR/OCr.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)

    train = x[:,:].reshape(-1, 400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.repeat(k, 500)[:, np.newaxis]

    np.savez('img/OCR/digits_for_ocr.npz', train = train, train_labels = train_labels)
    print('Save Data!')
# End of learningDigit()


def loadLearningDigit(ocrdata):
    # 학습한 내용이 저장된 파일을 열어 내용을 읽은 후, traindata와 traindata_labels를 리턴
    with np.load(ocrdata) as f:
        traindata = f['train']
        traindata_labels = f['train_labels']

    return traindata, traindata_labels
# End of loadLearningDigit()


def OCR_for_Digits(test, traindata, traindata_labels):
    # test인자는 인식할 손글씨 이미지를 resize20 함수로 처리한 리턴값
    # kNN을 이용해 가장 일치하는 결과를 도출하고 리턴 함.
    knn = cv2.ml.KNearest_create()
    knn.train(traindata, cv2.ml.ROW_SAMPLE, traindata_labels)
    ret, result, neighbors, dist = knn.findNearest(test, k=5)

    return result
# End of OCR_for_Digits()


def OCR_main():

    #learningDigit()
    # 초기 학습 후, digits_for_ocr.npz 생성




    ocrdata = 'img/OCR/digits_for_ocr.npz'
    traindata, traindata_labels = loadLearningDigit(ocrdata)
    digits = ['img/OCR/' + str(x) + '.jpg' for x in range(10)]

    print('traindata.shape = ', traindata.shape)
    print('traindata_labels.shape = ', traindata_labels.shape)

    savenpz = False
    for digit in digits:
        test = resize20(digit)
        result = OCR_for_Digits(test, traindata, traindata_labels)
        print('결과 = ', result)

        k = cv2.waitKey(0) & 0xFF
        # 인식결과가 틀리면 정확한 값을 키보드로 입력 -> 재 학습
        # 0~9 이외의 키를 누르면 재학습 하지않고 다음 For 구문으로 넘어감
        if k > 47 and k < 58:
            savenpz = True
            traindata = np.append(traindata, test, axis = 0)
            new_label = np.array(int(chr(k))).reshape(-1, 1)
            traindata_labels = np.append(traindata_labels, new_label, axis = 0)

    cv2.destroyAllWindows()

    if savenpz:
        print('재 학습!!')
        np.savez('img/OCR/digits_for_ocr.npz', train = traindata, train_labels = traindata_labels)

# End of OCR_main()







knn()
# 결과해석 k = 3
#   [[1.]] [[0.1.1]]
#   [[1.]] : 새로운 멤버 녹색원은 클래스 1에 속함(= 파란색 사각형에 속함)
#   [[0.1.1]] : 초록색원 주위에 가까운 이웃 멤버는 클래스0 멤버가 1개, 클래스 1 멤버가 2개

# k = 5로 했을때 결과
#   [[1.]] [[0. 1. 1. 0. 1.]]
#   [[1.]] : 새로운 멤버 녹색원은 클래스 1에 속함(= 파란색 사각형에 속함)
#   [[0. 1. 1. 0. 1.]] : 초록색원 주위에 가까운 이웃 멤버는 클래스0 멤버가 2개, 클래스 1멤버가 3개

OCR_main()
# 실행 순서
#   OCR_main()의 첫줄 learningDigit() 주석을 해제
#   learningDigit() 아래의 모든 코드를 주석 처리 후, 실행하면, digits_for_ocr.npz 파일을 생성 함.
#   다신 learningDigit() 주석처리 후, 아래 모든 코드 활성화 후 실행