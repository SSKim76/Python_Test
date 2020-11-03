"""
28강. 푸리에변환
    푸리에 변환
        푸리에 변환이란 주파수를 분석하는데 가장 많이 활용되는 도구
        주기를 가진 함수(주기함수)는 주파수가 다른 삼각함수의 합으로 표현 가능하다. 즉 주기는 무한대까지 가능하므로 "모든 함수는
        주파수가 다른 삼각함수의 합으로 표현 가능하다"라는 말...

        2D 이산 푸리에 변환(2D Discrete Fourier Transform; 2D-DFT)을 이미지에 적용하면 이미지를 주파수 영역으로 변환해 준다.
        DFT를 계산하기 위해서는 고속 푸리에 변환(Fast Fourier Trasnform; FFT)을 이용한다.

    Numpy를 활용한 푸리에 변환과 OpenCV를 이용한 푸리에 변환이 있다.

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef


def fourier(ImgNo):
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
    m_spectrum = 20*np.log(np.abs(fshift))
    # magnitude spectrum을 구한다.

    plt.subplot(121), plt.imshow(img, cmap ='gray')
    plt.title('input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(m_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()


def fourier_cv(ImgNo):
    img = cv2.imread(impDef.select_img(ImgNo))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    # 이미지의 2D DFT을 계산. 이미지를 이 함수의 인자로 입력시 반드시 np.float32로 랩핑해야함.
    # 이 함수는 복소수 형태(실수부, 허수부)로 결과를 리턴하는 것이 numpy를 이용할 때와의 차이점
    
    dft_shift = np.fft.fftshift(dft)
    m_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    # cv2.magnitude(x, y)
    # 2차원 벡터의 크기를 계산. 인자로 벡터의 x성분, y성분을 각각 입력

    plt.subplot(121), plt.imshow(img, cmap ='gray')
    plt.title('input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(m_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()


#fourier(0)
#fourier(28)
#fourier(12)

fourier_cv(0)
fourier_cv(28)
fourier_cv(12)


