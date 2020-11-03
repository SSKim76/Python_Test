
import cv2 as cv2

def select_img(sel_num):
    return{
        0 : 'img/100.jpg',                 # 경현
        1 : 'img/images_1.jpg',            # 그라데이션 이미지
        2 : 'img/multi_light.jpg',         #
        3 : 'img/opencv_logo.jpg',
        4 : 'img/sample_img1.jpg',         # 산과 강이 있는 풍경
        5 : 'img/sample_img2.jpg',         # 수지 이미지
        6 : 'img/sdoku.jpg',                #
        7 : 'img/tile.jpg',                 # Threshold Test용 타일
        8 : 'img/girl.png',                 # 기본여자 이미지
        9 : 'img/alpha1.jpg',               # 알파벳 및 숫자 이미지
        10: 'img/Code_Sample.jpg',          # Data Matrix Code Sample
        11: 'img/DW_Fail_img.jpg',          # Vericode Fail Image
        12: 'img/TW.jpg',                   # 트와이스
        13: 'img/TW_NY.png',                # Template Image 나연
        14: 'img/TW_SN.png',                # Template Image 사나
        15: 'img/TW_JW.png',                # Template Image 정연
        18: 'img/contour18.png',            # 사각형에 안으로 파인이미지
        19: 'img/contour19.png',            # 나무잎 비슷한, 별모양 비슷
        20: 'img/convexhull.png',           # 손으로 그린 손바닥?
        21: 'img/korea.png' ,               # 대한민국 지도
        22: 'img/star.jpg',                 # 별모양
        23: 'img/Histogram23.jpg',          # 히스토그램용 풍경
        24: 'img/24.jpg',                   # 히스토그램 균일화용 풍경
        25: 'img/25.jpg',                   # 히스토그램용 조각상
        26: 'img/26.jpg',                   # 히스토그램용 풍경
        28: 'img/28_BW_girl.jpg',           # 푸리에 Test용 이미지
        30: 'img/31.jpg',                   # Template Matching용 너구리 이미지
        31: 'img/bam.png',                  # Template Image 밤
        32: 'img/sdoku_1.jpg',              # 허프변환 Test용 스도쿠 이미지
        33: 'img/33.jpg',                   # 허프변환 Test용 원찾기
        34: 'img/34.jpg',                   # Watershed용 Test이미지(동전)
        35: 'img/35.jpg',                   # GrabCut용 Test Image
        37: 'img/37_corner.png',            # Corner 검출용 Image
        38: 'img/38_butterfly.png',         # SIFT Test용 나비
        42: 'img/42_ORB.png.',              # ORB용 Image - 암호와 해킹
        43: 'img/43_ORB.png',               # ORB용 회전 Image - 암호와 해킹
        44: 'img/43_1.png',                 # 이미지매칭용 등대가 있는 마을 사진
        45: 'img/43_2.png',                 # 이미지매칭용 액자를든 소녀 사진
        100: 'img/Test_Vericode.bmp'        # Vericode 그라데이션 Test용
    }.get(sel_num,'img/girl.png')


def close_window():
    cv2.waitKey(0)
    cv2.destroyAllWindows()