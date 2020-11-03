"""
22강. Image Contour 응용5

    Contour Hierarchy
        cv2.findContours()함수는 이미지에서 찾은 contour와 이 contour들의 hierarchy(계층)를 리턴함.
        contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APROX_SIMPLE)
        Contour Hierachy는 Contour의 부모형제관계를 나타내는 것

        cv2.findContours()함수의 두번째 리턴값은 Contour간의 관계를 나타냄.
        [Next, Previous, First Child, Parent]
            Next : 동일레벨의 다음 Contour 인덱스, 동일레벨의 다음 Contour가 없으면 -1
            Previous : 동일레벨의 이전 Contour 인덱스, 동일레벨의 이전 Contour가 없으면 -1
            First Child : 최초의 자식 Contour 인덱스, 자식 Contour가 없으면 -1
            Parent : 부모 Contour 인덱스, 부모 Contour가 없으면 -1

         cv2.RETR_LIST : 이미지에서 발견한 모든 Contour들을 계층에 상관하지 않고 나열
         cv2.RETR_TREE : 이미지에서 발견한 모든 Contour들의 관계를 명확히 해서 리턴턴
         cv2.RETR_EXTERNAL
         cv2.RETR_CCOMP






""






