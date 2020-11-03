"""
29강. 푸리에변환 응용
    푸리에 변환은 이미지를 주파수 영역으로 전환하여 이미지 프로세싱 작업을 수행할 수 있게 해주는 도구이고,
    주파수 영역에서 작업이 끝나면 역푸리에변환(Inversion Fourier Transform: IFT)을 수행하여 원래 이미지 영역으로
    되돌려서 이미지 프로세싱 결과를 확인 할 수 있음

    12강 참조 LPF(Low Pass Filger) & HPF(Hight, Pass Filter)
        LPF를 적용하면 낮은 주파수 대역만 남아 있는 이미지가 되어 블러(blur)효과를 가진 이미지가 됨
        HPF를 적용하면 높은 주파수 대역만 남아 있는 이미지가 되어 사물의 경계나 노이즈 등 만이 남아 있는 이미지가 됨
        푸리에 변환이란 주파수를 분석하는데 가장 많이 활용되는 도구
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef



# 이미지를 푸리에 변환하여 주파수 영역으로 전환하고, 주파수 영역 이미지 중앙에 A x A크기의 정사각형 영역에 해당하는
# 값을 모두 0으로 변환, 즉 HPF를 적용한 것과 같은 결과가 도출 됨
# 다시 역푸리에 변환을 이용해 원래 이미지 영역으로 전환하면 주파수가 낮은 부분이 없는 새로운 이미지가 됨.
def fourier(ImgNo, SIZE=30):
    img = cv2.imread(impDef.select_img(ImgNo))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(img)
    # image의 2D DFT를 계산

    fshift = np.fft.fftshift(f)
    # 2D DFT를 계산하여 얻어진 푸리에 변환 결과는 주파주가 0인 컴포넌트를 좌상단에 위치시킨다.
    # np.fft.fftshift(f)는 주파수가 0인 부분을 정 중앙에 위치시키고 재 배열해주는 함수이다.
    """
        >>> import numpy as np
        >>> f = np.fft.fftfreq(10, 0.1)
        >>> f
            [0. 1. 2. 3. 4. -5. -4. -3. -2. -1.]
        >>> fshift = np.fft.fftshift(f)
        >>> fshift
            [-5. -4. -3. -2. -1. 0. 1. 2. 3. 4.]
    """

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)

    fshift[crow-(int(SIZE/2)):crow+int((SIZE/2)), ccol-int((SIZE/2)):ccol+int((SIZE/2))] = 0
    # 주파수 영역의 이미지 정중앙에 SIZE x SIZE 크기의 영역에 있는 모든값을 0으로 만든다.

    #역 푸리에변환
    f_ishift = np.fft.ifftshift(fshift)
    # 역 쉬프트 함수 재 배열된 주파수 값을의 위치를 원래대로 되돌린다.

    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # 역푸리에 변환을 하여 원래 이미지 영역으로 전환한 후, 모든 값에 절대값을 취한다.


    plt.subplot(131), plt.imshow(img, cmap ='gray')
    plt.title('input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(img_back, cmap = 'gray')
    plt.title('After HPF'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(img_back)
    plt.title('After HPF'), plt.xticks([]), plt.yticks([])

    plt.show()


fourier(0, 30)
fourier(28, 30)
fourier(12, 30)

fourier(10, 10)
fourier(10, 30)
fourier(10, 50)

fourier(11, 10)
fourier(11, 30)
fourier(11, 50)


