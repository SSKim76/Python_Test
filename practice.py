# print(5)
# print(-10)
# print(3.14) 
# print(1000)
# print(5*10)



# 문자형 자료
# print('풍선')
# print("나비")
# print("ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")
# print('ㅋ' * 9)



# Bool
# print(5 > 10)
# print(5 < 10)
# print(not True)
# print(not (5 > 10))



# 변수
# 애완동물을 소개해 주세요
# 선택 + Ctrl + / = 전체 주석
# print(" 우리집 강아지의 이름은 연탄 이예요")
# print(" 연탄이는 4살이며, 산책을 아주 좋아해요")
# print(" 연탄이는 어른일까요? True")

name = " 해피"
animal = "고양이"
hobby = " 공놀이"
age = 4
adult = age >= 3

# print(" 우리집 " + animal + "의 이름은 "+ name + "예요")
# print(name," 는 " , age,"살이며, ",hobby ,"을 아주 좋아해요")
# print(name + "는 어른일까요? " + str(adult))

# stations = ("사당", "신도림", "인천공항")
# for station in stations:
#     print(station+"행 열차가 들어오고 있습니다.")
# End of for station in stations:



# 연산자
# print(2**3)                                 # 2^3 = 8
# print("5/3의 나머지 구하기 = ", 5%3)         # 나머지 구하기 = 2
# print(5//3)                                 # 몫 구하기 = 2
# print(1 != 3)                               # 1과 3이 같지않다면 True
# print(not (1 !=3))                           # False
# print((3 > 0) and (3 < 5))                  # True
# print((3 > 0) & (3 < 5))                    # 위 문장과  동일문장
# print(( 3> 0) or (3 > 5))                   # True
# print(( 3> 0) | (3 > 5))                    # 위 문장과 동일문장
# print(5 > 4 > 3)                            # True



# 수식
# number = 2 + 3 * 4      # 14
# print("number = ", number)
# number += 2             # 16
# print("number = ", number)
# number *= 2             # 32 
# print("number = ", number)
# number /= 2             # 16
# print("number = ", number)
# number -= 2             # 14
# print("number = ", number) 



###########################################################################################
# # 숫자처리 함수
# print(abs(-5))          # 절대값 5
# print(pow(4,2))         # 4 x 4 = 16
# print(max(5,23,12))     # 최대값 23
# print(min(12, 13, 5))   # 최소값 5
# print(round(3.14))      # 반올림 3

# from math import *
# print(floor(4.99))      # 내림 4
# print(ceil(3.14))       # 올림 4
# print(sqrt(16))         # 제곱근 4



##############################################################################################
# 랜덤 함수
# from random import *
# print(random())                 # 0.0 ~ 1.0 미만의 난수 발생
# print(random() * 10)            # 0.0 ~ 10.0 미만의 난수 발생
# print(int(random() * 10))       # 정수형 만
# print(int(random() * 45 + 1))   # 1 ~ 45 이하의 임의의 값 생성
# print(randrange(1,46))          # 1 ~ 46 미만의 임의의 값 생성
# print(randint(1,45))            # 1 ~ 45 이하의 임의의 값 생성

# studyDay = randint(4, 28)
# print("오프라인 스터디 모임 날짜는 매월 " + str(studyDay)+"일로 선정 되었습니다!!")



##############################################################################################
# 문자열
# sentence = '나는 소년입니다.'
# print(sentence)
# sentence2 = " 파이썬은 쉬워요"
# print(sentence2)
# sentence3 = """
# 나는 소년이고,
# 파이썬은 쉬워요
# """
# print(sentence3)

# jumin = "990120-1234567"

# print("성별 : " + jumin[7])
# print("연도 : " + jumin[0:2])
# print("월 : " + jumin[2:4])
# print("일 : " + jumin[4:6])
# print("생년월일 : " + jumin[:6])    # 처음 부터 6 직전까지
# print("뒤 7자리 : " + jumin[7:])    # 8번재 부터 마지막 까지
# print("뒤 7자리(뒤에서 부터) : " + jumin[-7:-1])



##################################################################################################
# 문자열처리 함수
# python = "Python is Amaging"
# print(python)
# print(python.lower())                   # 소문자로 변환
# print(python.upper())                   # 대문자로 변환
# print(python[0].isupper())              # python변수의 첫번째 글자가 대문자이면 True를 반환
# print(len(python))                      # python변수의 글자 수
# print(python.replace('Python', 'Java')) # python변수내용 중 'Python'을 'Java'로 변환

# index = python.index("n")               # python변수에서 'n'의 위치를 찾는다.
# print(index)                            # 첫번째 'n'의 위치 5를 출력
# index = python.index('n', index + 1)    # 2번째 'n'을 찾는다.
# print(index)                            # 두번째 'n'의 위치 15를 출력
# print(python.find('n'))                 # python변수에서 'n'의 위치를 찾아서 출력 : 5
# print(python.find('Java'))              # python변수에서 'Java'의 위치를 찾아서 출력, 없으면 -1을 출력
# #print(python.index('Java'))             # python변수에서 'Java'의 위치를 찾아서 출력, 없으면 오류발생
# print(python.count('n'))                # python 변수에서 'n'이 몇번 등장하는지 출력 : 2



##############################################################################################
# 문자열 포맷
# print("a" + "b")
# print('a' + 'b')
# print('a', 'b')

# # 방법 1
# print("나는 %d살 입니다." %20)         # %d 정수값만 입력가능
# print("나는 %s을 좋아해요." % "파이썬") # %s 문자열 입력
# print("Apple은 %c로 시작해요." % 'A')   # %c 한 문자 입력

# print("나는 %s살 입니다." %20)         # %d 정수값만 입력가능
# print("나는 %s을 좋아해요." % "파이썬") # %s 문자열 입력
# print("Apple은 %s로 시작해요." % 'A')   # %c 한 문자 입력

# print("나는 %s색과 %s색을 좋아히요." % ("파란", "빨간"))

# # 방법2
# print("나는 {}살 입니다.".format(20))
# print("나는 {}색과 {}색을 좋아해요.".format("파란", "빨간"))
# print("나는 {1}색과 {0}색을 좋아해요.".format("파란", "빨간"))

# # 방법3
# print("나는 {age}살이고, {color}색을 좋아해요.".format(age = 20, color = "노란"))

# # 방법4(Version 3.6이상 사용가능)
# age = 20
# color = "보라"
# print(f"나는 {age}살이고, {color}색을 좋아해요.")   # f뒤에 공백이 있으면 안됨.



################################################################################################3
# 탈출문자
# # \n : 줄바꿈
# print("백문이 불여일견\n백견이 불여일타")

# # \" , \' : 문자열을 그대로 출력
# print('저는 "나도코딩" 입니다.')
# print("저는 \"나도코딩\" 입니다.")

# # \\ : 문장내에서 \ 하나만 출력
# print("c:\\Python_Test\\practice.py ")

# # \r : 커서를 맨앞으로 이동
# print("Red Apple\rPine")    # Red Apple을 출력 후, 커서를 맨 앞으로 이동하여 Red_을 Pine로 덮어 씀 

# # \b : 백스페이스(한 글자 지우기)
# print("Redd\bApple")        # Redd출력 후, 마지막 d를 지운 후(Red), Apple를 출력(RedApple) 

# # \t : 탭
# print("Red \t Apple")


###############################################################################################
#퀴즈 #3
# 사이트별로 비밀번호를 만들어 주는 프로그램을 작성하시오.
# 예) http://naver.com
# 규칙 1 : http:// 부분은 제외 => naver.com
# 규칙 2 : 처음만나는 점(.) 이후 부분은 제외 => naver만 사용
# 규칙 3 : 남은 글자 중 처음 세자리(nav) + 남은 글자수(5) + 남은 글자내 'e' 갯수(1) + '!'로 구성
# 예) 생성된 비밀번호 : nav51!

# url = "http://naver.com"
# url = "http://google.com"
# # http:// 제거
# newUrl = url.replace("http://", "")
# print(newUrl)
# newUrl2 = newUrl[:newUrl.find(".")]
# print(newUrl2)
# cutUrl = newUrl2[:3] 
# urlCnt = len(newUrl2)
# eCnt = newUrl2.count('e') + 5
# print(f"cutUrl = {cutUrl}, urlCnt = {urlCnt}, eCnt = {eCnt}")
# newPass = cutUrl+str(urlCnt)+str(eCnt)+"!"
# print("{ }의 생성된 비밀번호 : {}".format(url, newPass))



################################################################################################# 
# List[]

# # 지하철 칸별로 10명, 20명, 30명
# subway1 = 10
# subway2 = 20
# subway3 = 30

# subway = [10, 20, 30]
# print(subway)

# subway = ['유재석', '조세호', '박명수']
# print(subway)

# # 조세호씨가 몇 번째 칸에 타고 있는가?
# print(subway.index('조세호'))

# # 하하씨가 다음 정류장에서 다음칸에 탐
# subway.append("하하")
# print(subway)

# # 정형돈씨를 유재석씨와 조세호씨 사이에 태움
# subway.insert(1, "정형돈")
# print(subway)

# # 뒤에서 부터 1명씩 꺼냄
# print(subway.pop())
# print(subway)

# # print(subway.pop())
# # print(subway)

# # print(subway.pop())
# # print(subway)

# # 같은 이름의 사람이 몇명 있는지 확인
# subway.append("유재석")
# print(subway)
# print(f"유재석씨는 몇명 일까? : {subway.count('유재석')} 명")
# print("유재석씨는 몇명 일까? : {} 명".format(subway.count("유재석")))


# #정렬
# numList = [1,5,6,87,4,2]
# print("정렬 전 : {}".format(numList))
# numList.sort()
# print(f"정렬 후 : {numList}")


# # 뒤집기
# numList.reverse()
# print(f"거꾸로 정렬 후 : {numList}")


# # 모두 지우기
# #numList.clear()
# #print(f"모두 지운 후 : {numList}")


# # 다양한 자료형 함께 사용 가능
# mixList = ["조세호", 20, True]
# print(f"mixList : {mixList}")


# # 리스트 합치기
# numList.extend(mixList)
# print(f"numList : {numList}")


# # 이건 안되는가 봄???
# # sumList = []
# # print("sumList : {}".format(sumList))
# # sumList.extened(mixList, numList)
# # print("sumList : {}".format(sumList))



#################################################################################################
# 사전{key:value로 저장}
# key는 중복이 안됨!
# cabinet = {3:"유재석", 100:"김태호"}
# print("cabinet = {}".format(cabinet))
# key = 3
# print("{}번 키의 값은 = {}".format(key, cabinet[key]))

# for key in cabinet:
#     value = cabinet[key]
#     print("{}번 키의 값은 = {}".format(key, value))
#     print("{}번 키의 값은 = {}".format(key, cabinet[key]))

# print("cabinet.get(100) = {}".format(cabinet.get(100)))
# #print(cabinet[5])                         # [] Key Error 발생시 프로그램 중단!!
# print(cabinet.get(5))                       # Key Error 발생해도 계속 실행 None 출력 => get사용 추천!!
# print(cabinet.get(5, "Key가 사용가능 함.")) # Key Error 발생시 None 대신 "출력할 문장"안에 문자를 출력할 수 있음

# # Key 가 있는지 확인할 때
# print(f"print(5 in cabinet) = {5 in cabinet}")  # 없으면 False
# print(f"print(3 in cabinet) = {3 in cabinet}")  # 있으면 True

# cab = {"A3":"유재석", "B100":"김태호"}
# print(f"cab.get(A3) = {cab.get('A3')}")  
# print(f"cab.get(B100) = {cab.get('B100')}") 


# # 수정 및 추가
# cab["C20"] = "조세호"
# cab["A3"] = "김종국"
# print(f"수정 및 추가 후 cab = {cab}")


# # 삭제
# del cab["A3"]
# print(f"삭제 후 cab = {cab}")


# # 현재 Key 값만 출력
# print(f"현재 사용중인 Key 값을 출력 = cab.keys() : {cab.keys()}")


# # 현재 value 값만 출력
# print(f"현재 사용중인 value 값을 출력 = cab.values() : {cab.values()}")


# # 현재 Key, value 값을 쌍으로 출력
# print(f"현재 사용중인 Key, value 값을 출력 = cab.items() : {cab.items()}")


# # 모든 값 지우기 
# cab.clear()
# print(f"현재 사용중인 Key, value 값을 삭제 = cab.clear() : {cab }")




###############################################################################################
# 튜플(내용변경 및 추가불가, 단 리스트보다 처리속도가 빠름!)
# menu = ("돈까스", "치즈까스")
# print(menu)
# print(menu[0])

# # menu.add("생선까스") add를 지원하지 않음

# name = "김종국"
# age = 20
# hobby = "코딩"
# print(name, age, hobby)

# (name, age, hobby) = ("유재석", 30, "독서")
# print(name, age, hobby)

# name, age, hobby = "이수근", 25, "수영"
# print(name, age, hobby)




#################################################################################################
# # 세트(set) 집합
# # 중복 안됨, 순서 없음
# my_set = {1,2,3,3,3}
# print(my_set)   # {1,2,3} 으로 출력 

# java = {"유재석", "김태호", "정형돈"}
# python = set({"유재석", "박명수"})


# # 교집합(Java와 Python을 모두 할 수 있는 개발자)
# print(java & python)
# print(java.intersection(python))    


# # 합집합(java를 할 수 있거나, Python을 할 수 있는 개발자)
# print(java | python)
# print(java.union(python))


# # 차집합(java가능, Python을 못하는 개발자)
# print(java - python)
# print(java.difference(python))


# # python을 할 줄 하는 사람이 늘어남
# python.add("김태호")
# print(python)


# # java를 잊어버렸어요
# java.remove("김태호")
# print(java)


# 자료구조의 변경
# 커피숍
# menu = {'커피', '우유', '주스'}
# print(menu, type(menu))

# menu = list(menu)
# print(menu, type(menu))

# menu = tuple(menu)
# print(menu, type(menu))

# menu = set(menu)
# print(menu, type(menu))



##############################################################################################
# # 퀴즈 #4
# # 당신의 학교에서는 파이썬 코딩 대회를 주최합니다.
# # 참석률을 높이기 위해 댓글 이벤트를 지행하기로 하였습니다.
# # 댓글 작성자들 중에 추첨을 통해 1명은 치킨, 3명은 커피 쿠폰을 받게 됩니다.
# # 추첨 프로그램을 작성하시오.

# # 조건 1 : 편의상 댓글은 20명이 작성하였고, 아이디는 1 ~ 20 이라고 가정
# # 조건 2 : 댓글 내용과 상관 없이 무작위로 추첨하되 중복불가
# # 조건 3 : random 모듈의 shuffle과 sample을 활용

# # 출력예제
# # --- 당첨자 발표 ---
# # 치킨 당첨자 : 1
# # 커피 당첨자 : [2, 3, 4]
# # --- 축하합니다. ---

# # 활용예제
# # from random import *
# # lst = [1,2,3,4,5]
# # print(lst)
# # shuffle(lst)
# # print(lst)
# # print(sample(lst, 1)) # lst중에서 랜덤으로 1개를 추출
# # print(sample(lst, 4)) # lst 중에서 랜덤으로 4개 추출

# from random import *
# users = list(range(1,21))
# print(users)
# shuffle(users)
# print(users)
# choiceUser = sample(users,4)
# print(choiceUser)
# print(choiceUser[0])
# print(choiceUser[1], choiceUser[2], choiceUser[3])
# print("--- 당첨자 발표 ---")
# print("치킨 당첨자 : {}".format(choiceUser[0]))
# print(f"커피 당첨자 : {choiceUser[1]}, {choiceUser[2]}, {choiceUser[3]} ")
# print("커피 당첨자 : {}".format(choiceUser[1:])) 
# print("--- 축하합니다. ---")



###################################################################################################
# if
# input("문장")  => 사용자의 입력을 string형으로 받는다.
# if 조건문1 : 
#   조건문1이 True일 경우 실행하고 빠져나감
# elif 조건문2 : 
#   조건문2가 True일 경우 실행하고 빠져나감
# else : 
#   조건문1, 2모두 False일 경우 실행하고 빠져나감



####################################################################################################
# for
# for no in [0,1,2,3,4]:    # 0,1,2,3,4
# for no in range(6):       # 0,1,2,3,4
# for no in range(1,6):     # 1,2,3,4,5
#   print("대기번호{0}".format(no))
#
# starbucks = ["아이언맨", "토르", "스파이더맨"]
# for customer in starbucks:
#   print("{0}님, 커피가 준비되었습니다.".format(customer))



####################################################################################################
# while
# while(조건): # 조건이 만족하는 동안 반복
# customer = "토르"
# index = 5
# while index >= 1:
#   print("{0}님, 커피가 준비 되었습니다. {1}번 남았어요".format(customer, index))
#   index -= 1
#



#####################################################################################################
# continu 와 break
# continu : 다음 반복문으로 넘어감. 문장을 실행하지 않고 건너 뜀(skip)
# break : 더이상 반복을 진행하지 않고 종료.



##################################################################################################
# # 한줄 for
# # 출석번호가 1,2,3,4,5 인 학교에서, 번호를 101,102,103,104,105로 앞에 100을 붙이기로 함
# student = [1,2,3,4,5]
# print(student)
# student = [i+100 for i in student]
# print(student)


# # 학생 이름을 길이로 변환
# students = ["Iron Man", "Thor", "I am groot"]
# #students = [len(i) for i in students]
# print(students)


# # 학생 이름을 대문자로 변환
# students = [i.upper() for i in students]
# print(students)



#####################################################################################################
# 퀴즈 5
# 당신은 Cocoa 서비스를 이용하는 택시 기사님 입니다.
# 50명의 승객과 매칭 기회가 있을 때, 총 탑승 승객 수를 구하는 프로그램을 작성하세요.

# 조건 1 : 승객별 운행 소요시간은 5분 ~ 50분 사이의 난수로 정해짐.
# 조건 2 : 당신은 소요시간 5분 ~ 15분 사이의 승객만 매칭해야 합니다.

# (출력문 예제)
# [O] 1번째 손님 (소요시간 : 15분)
# [X] 2번째 손님 (소요시간 : 50분)
# [O] 3번째 손님 (소용시산 : 5분)
# ......
# [X] 50번째 손님 (소요시간 : 16분)

# 총 탑승 승객 : 2분

# from random import *

# minTime = 5
# maxTime = 50
# my_minTime = 5
# my_maxTime = 15
# cnt = 0

# for i in range(1,51):
#     takeTime = randrange(minTime, maxTime+1)
#     if  my_minTime <= takeTime <= my_maxTime:
#         cnt += 1
#         icon = "O"
#     else:
#         icon = " "

#     print("[{0}] {1}번째 손님 (소요시간 : {2}분)".format(icon, i, takeTime))
# print("총 탑승 승객 : {0} 분".format(cnt))




######################################################################################################
# 함수(Function)
# def open_account():
#     print("새로운 계좌가 생성되었습니다.")

# def deposit(balance, money):
#     print("입금 {0}원 이 완료 되었습니다. 잔액은 {1} 원 입니다. ".format(money, balance+money))
#     return(balance+money)

# def withdraw(balance, money):
#     if balance >= money:
#         balance = balance - money
#         print("출금이 완료 되었습니다. 현재 잔액은 {0}원 입니다.".format(balance))        
#     else:
#         print("잔액이 부족합니다. 현재 잔액은 {0}원 입니다.".format(balance))

#     return balance

# def withdraw_night(balance, money):
#     commission = 100

#     if balance >= money:
#         balance = balance - money - commission
#         print("출금 {0}원 과 수수료 {1}원 이 처리 되었습니다. 현재 잔액은 {2}원 입니다." \
#             .format(money, commission, balance))        
#     else:
#         print("잔액이 부족합니다. 현재 잔액은 {0}원 입니다.".format(balance))

#     return commission, balance

# balance = 0
# open_account()

# balance = deposit(balance, 1000)
# #print("balance 입금 후, 현재 잔액은 = {0}".format(balance))

# balance = withdraw(balance, 300)
# #print("balance 출금 후, 현재 잔액은 = {0}".format(balance))

# commission, balance = withdraw_night(balance, 300)
# print(f"commission = {commission}, balance = {balance}")

# def profile(name, age, main_Lang):
#     print("이름 : {0} \t 나이 : {1}\t 주 사용 언어 : {2}".format(name, age, main_Lang))

# profile("유재석", 20, "파이썬")
# profile("김태호", 25, "자바")

# 가정1 : 같은 학교 같은 학년 같은 반 같은 수업
# def profile(name, age = 17, main_Lang = "파이썬"):
#     print("이름 : {0} \t 나이 : {1}\t 주 사용 언어 : {2}".format(name, age, main_Lang))

# profile("유재석")
# profile("김태호")
# profile("정형돈", 16, "자바")
# profile(name = "박명수", main_Lang = "자바", age = 18 )


# def profile(name, age , lang1, lang2, lang3, lang4, lang5):
#     print("이름 : {0} \t 나이 : {1}\t".format(name, age), end = " ") # end = " " 줄바꿈을 하지 않음
#     print(lang1, lang2, lang3, lang4, lang5)

# profile("유재석", 20, "Python", "java", "C", "C++", "C#")
# profile("김태호", 25, "Kotlin", "Swift", "","","")
# #profile("박명수", 28, "VB")     # 오류발생


# def profile1(name, age , *language):     # *language = 가변인자
#     print("이름 : {0} \t 나이 : {1}\t".format(name, age), end = " ") # end = " " 줄바꿈을 하지 않음
#     for lang in language:
#         print(lang, end = " ")
#     print()

# profile1("유재석", 20, "Python", "Java", "C", "C++", "C#", "JavaScript")
# profile1("김태호", 25, "Kotlin", "Swift", "","","")
# profile1("박명수", 28, "VB")




######################################################################################################
#지역변수와 전역변수
# gun = 10

# def checkpoint(soldiers) : # 경계근무
#     #gun = 20
#     global gun          # 메모리 & 관리상 문제로 global 잘 사용하지 않음...
#     gun = gun - soldiers
#     print("[함수내] 남은 총 : {0}".format(gun))


# # 추천하는 방법
# def checkpoint2(gun, soldiers):
#     gun = gun - soldiers
#     print("[함수내] 남은 총 : {0}".format(gun))

#     return gun 


# print("[함수외부] 전체 총 : {0} ".format(gun))
# checkpoint(2)   # 2명이 총을 들고 근무나감
# print("[함수외부] 전체 총 : {0} ".format(gun))
  
# print("[함수외부] 전체 총 : {0} ".format(gun))
# gun = checkpoint2(gun, 3)   # 2명이 총을 들고 근무나감
# print("[함수외부] 전체 총 : {0} ".format(gun))   



#############################################################################################
# # ## 퀴즈 6
# # 표준 체중을 구하는 프로그램을 작성하시오.

# # * 표준체중 : 각 개인의 키에 적당한 체중

# # (성별에 따른 공식)
# # 남자 = 키(m) x 키(m) x 22
# # 여자 = 키(m) x 키(m) x 21

# # 조건 1 : 표준 체중은 별도의 함수 내에서 계산
# #     * 함수명 : std_weight
# #     * 전달값 : 키(height), 성별(gender)
# # 조건2 : 표준 체중은 소수점 둘째자리까지 표시

# # (출력예제)
# # 남자 키 175Cm의 표준체중은 67.38Kg 입니다.

# def std_weight(height, gender):
#     gen = ""
#     if gender == "M": # 남자
#         std = 22
#         gen = "남자"
#     else :  # 여자
#         std = 21
#         gen = "여자"
    
#     std_weight = round((height * height)/10000 * std, 2)
#     print("{0} 키 {1}Cm의 표준체중은 {2}Kg 입니다.".format(gen, height, std_weight))


# std_weight(175, "M")
# std_weight(160, "F")




#############################################################################################
# #   표준 입출력 : 2:59:00

# # sep = ","  등
# print("Python", "Java")
# print("Python" , "Java", sep = ",")  
# print("Python" , "Java", sep = " vs ")  

# # end = "?" 문장의 끝 부분을 줄바꿈에서 ?로 바꿔줘
# print("Python" , "Java", sep = ",", end = "?")
# print(" 무엇이 더 재미있을까요?")  

# import sys
# print("Python", "Java", file = sys.stdout)  # 표준출력으로 문장을 출력
# print("Python", "Java", file = sys.stderr)  # 표준에러로 문장을 출력 

# # 시험성적
# #.ljust(8), .rjust(8) : 8칸의 공간을 확보 후, 왼쪽정렬(.ljust(8)), 오른쪽정력(.rjust(8))
# scores = {"수학" : 0, "영어":50, "코딩":100}
# for subject, score in scores.items():
#     #print(subject, score, end = " ")
#     print(subject.ljust(8), str(score).rjust(4), sep =":")  # .ljust(8) 8칸의 공간을 확보 후, 왼쪽으로 정렬

# # .zfill(3) 3자리로 만들고 빈칸은 0으로 채움
# #은행 대기순번표
# # 001, 002, 003, ......
# for num in range(1,21):
#     print("대기번호 : "+str(num).zfill(3))


# # 표준입력 
# answer = input("아무 값이나 입력하세요 : ")     # 문자열 형태로 입력 받음
# print("입력하신 값은 " + answer + "  입니다.")



##################################################################################################
# # 다양한 출력포멧 : 3:10:12

# # 빈 자리는 빈 공간으로 두고, 오른쪽 정렬을 하되, 총 10자리 공간을 확보
# print("{0: >10}".format(500))

# # 양수일 땐 +표시, 음수일 땐 -표시
# print("{0: >+10}".format(500))
# print("{0: >+10}".format(-500))

# # 왼쪽 정렬을 하고, 빈칸은 _로 채움
# print("{0:_<10}".format(500))

# # 3자리 마다 콤마 찍어주기
# print("{0:,}".format(100000000))

# # 3자리 마다 콤마 찍고, 부호표시
# print("{0:+,}".format(100000000))   # + 이후에 , 반대로 , 다음에 + 할경우 에러 발생
# print("{0:+,}".format(-100000000))

# # 3자리 마다 콤마 찍고, 부호표시, 자릿수 확보하고 빈자리는  ^로 채우기
# print("{0:^<+30,}".format(100000000000))

# # 소수점 출력
# print("{0:f}".format(5/3))
# # 소수점 3째 자리에서 반올림 하여  출력
# print("{0:.2f }".format(5/3))



##############################################################################################
# 파일 입출력 : 3:17:48

# # score.txt 파일을 쓰기모드로 열기(덮어쓰기!). 한글 사용가능하도록(utf8)
# score_file = open("score.txt", "w", encoding="utf8") 

# # score.txt 파일에 쓰기(print 사용)
# print("수학 : 0", file = score_file)
# print("영어 : 50", file = score_file)

# # 파일 사용후에는 꼭! 닫아 준다!!
# score_file.close()

# # 파일을 추가 모드로 연다
# score_file = open("score.txt", "a", encoding="utf8")
# # print 문은 자동으로 줄바꿈이 되지만 .write는 자동 줄바 꿈이 없음... 그래서 \n을 추가해야함.
# score_file.write("과학 : 80")
# score_file.write("\n코딩 : 100")
# score_file.close()

# # 읽기모드로 오픈 한꺼번에 읽어오기
# score_file = open("score.txt", "r", encoding="utf8")
# print(score_file.read())       # .read() : 파일의 모든 내용을 읽어 옴 
# score_file.close()


# # 읽기모드로 오픈 한줄씩 출력
# score_file = open("score.txt", "r", encoding="utf8")
# print(score_file.readline(), end="")       # .readline() : 줄별로 이동, 한 줄일고 커서는 다음 줄로 이동
# print(score_file.readline(), end="")
# print(score_file.readline(), end="")
# print(score_file.readline())
# score_file.close()

# # 몇줄인지 모르는 경우 읽어서 출력하는 법, 반복문 사용
# # score_file = open("score.txt", "r", encoding="utf8")
# # while True:
# #     line = score_file.readline()
# #     if not line:
# #         break
# #     print(line, end="")
# # score_file.close()

# # list에 값을 넣어 처리하는 방법
# score_file = open("score.txt", "r", encoding="utf8")
# lines = score_file.readlines()      # 모든 라인을 list 형태로 저장
# for line in lines:
#     print(line, end = "")
# score_file.close()




########################################################################################################
# pickle : 3:26:26
# 프로그램상에서 사용하고 있는 데이터를 파일로 저장하는 기능
# 피클은 꼭 바이너리 파일로 저장
# 따로 인코딩 설정은 필요 없음
 
# import pickle
# # profile_file = open("profile.pickle", "wb")    # wb = 바이너리파일로 쓰기모드
# # profile = {"이름":"박명수", "나이":30, "취미":["축구", "골프", "코딩"]}

# # print(profile) # dump 전 내용확인 용 출력
# # pickle.dump(profile, profile_file)      # profile에 있는 정보를 profile_file에 저장
# # profile_file.close()

# profile_file = open("profile.pickle", "rb") # 바이너리형태로 읽기
# profile = pickle.load(profile_file)     # 파일에 있는 정보를 profile로 읽어오기
# print(profile)
# profile_file.close()



####################################################################################################
# with : 3:30:23
# 파일처리를 조금 더 쉽게 할 수 있음
# 별도의 .close()가 필요없음
#   

# import pickle
# with open("profile.pickle", "rb") as profile_file : # profile.pickle를 열어서 profile_file이라는 변수에 저장
#     print(pickle.load(profile_file))        # pickle.load를 통해서 profile_file을 연다, 별도의 .close()가 필요없음

# with open("study.txt", "w", encoding="utf8") as study_file:
#     study_file.write("파이썬을 열심히 공부하고 있어요 ^^")

# with open("study.txt", "r", encoding="utf8") as study_file:
#     print(study_file.read())




#######################################################################################################
## 퀴즈 7

# 당신의 회사에서는 매주 1회 작성해야 하는 보고서가 있습니다.
# 보고서는 항상 아래와 같은 형태로 출력되어야 합니다.

# - x 주차 주간보고 - 
# 부서 : 
# 이름 : 
# 업무요약 : 

# 1주차 부터 50주차 까지의 보고서 파일을 만드는 프로그램을 작성하시오.
# 조건 : 파일명은 '1주차.txt', '2주차.txt', .... 와 같이 만듭니다.


# # 첫 번째 방법 OK
# for week in range(1, 51):
#     fileName = str(week)+"주차.txt"
#     contents = "- "+str(week)+" 주차 주간보고 - \n 부서 : \n 이름 : \n 업무요약 : "
#     with open(fileName, "w", encoding="utf8") as makeFile:
#         makeFile.write(contents)


# # 두 번째 방법 OK
# for week in range(1, 51):
#     fileName = str(week)+"주차.txt"
#     openFile = open(fileName, "w", encoding="utf8")
#     print("- {0} 주차 주간보고 - ".format(week), file = openFile)
#     print("부서 : ", file = openFile)
#     print("이름 : ", file = openFile)
#     print("업무요약 : ", file = openFile)
#     openFile.close()





##########################################################################################################
#
#    class : 3:38:08
#

# def make_unit_print(name, hp, damage):
#     print("\n{0} 유닛이 생성 되었습니다.".format(name))
#     print("체력 : {0}, 공격력 : {1}\n ".format(hp, damage))


# def attack(name, location, damage):
#     print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 : {2}]".format(name, location, damage))



# # 마린 : 공격 유닛, 군인, 총을 쏠 수 있음
# name = "마린"       # 유닛의 이름
# hp = 40             # 유닛의 체력
# damage = 5          # 유닛의 공격력
# print("\n{0} 유닛이 생성 되었습니다.".format(name))
# print("체력 : {0}, 공격력 : {1}\n ".format(hp, damage))


# # 탱크 : 공격 유닛, 탱크, 포를 쏠 수 있음, 일반모드와 시즈모드가 있음.
# tank_name = "탱크"
# tank_hp = 150
# tank_damage = 35
# make_unit_print(tank_name, tank_hp, tank_damage)

# # 탱크가 1대 추가될 경우
# tank2_name = "탱크"
# tank2_hp = 150
# tank2_damage = 35
# make_unit_print(tank2_name, tank2_hp, tank2_damage)

# # 공격
# attack(name, "1시", damage)
# attack(tank_name, "1시", tank_damage)
# attack(tank2_name, "1시", tank2_damage)



# 유닛이 무한정 생성될 경우.. 위의 방법으로 사용 불가 ... 그래서 class로 사용
# Class 생성............
# 일반 유닛
# class Unit:
#     def __init__(self, name, hp, damage):
#         self.name = name
#         self.hp = hp
#         self.damage = damage
#         print("\n{0} 유닛이 생성 되었습니다.".format(self.name))
#         print("체력 : {0}, 공격력 : {1}\n ".format(self.hp, self.damage))

# # 사용..............
# marine1 = Unit("마린", 40, 5)
# marine2 = Unit("마린", 40, 5)
# tank = Unit("탱크", 150, 35)



########################################################################
# 
# class : __init__  : 3:47:04
# 생성자... 자동호출객체
# 객체 : 클래스로 만들어진 녀석들... 마린... 탱크.. (마린과 탱크는 Unit 클래스의 인스턴스라고 부름)




########################################################################
# 
# class : 멤버변수  : 3:48:34
# 멤버변수 : 클래스 안에서 정의된 변수(self.name, self.hp, self.damage)
# class 외부에서 멤버 변수를 추가(확장) 할 수도 있음
# 주의 : 외부에서 추가(확장)된 멤버변수는 확장한 객체에서만 사용가능, 다른 객체에서는 사용불가 


# # 레이스 : 공중유닛, 비행기, 클로킹(상대방에게 보이지 않음)
# wraith1 = Unit("레이스", 80, 5)

# # 클래스 외부에서 멤버변수 사용가능
# print("유닛이름 : {0}, 체력 : {1}, 공격력 : {2},".format(wraith1.name, wraith1.hp, wraith1.damage))

# #마인드 컨트롤 : 상대방 유닛을 내 것으로 만드는 것(빼앗음)
# wraith2 = Unit("빼앗은 레이스", 80, 5)

# # 외부에서 멤버변수를 추가 할 수 있음, 
# wraith2.clocking = True

# if wraith2.clocking:
#     print("{0} 은(는) 현재 클로킹 상태 입니다.\n".format(wraith2.name))

# # 외부에서 추가(확장)된 멤버변수는 확장한 객체에서만 사용가능, 다른 객체에서는 사용불가
# # if wraith1.clocking:        # Error 발생, clocking은 wraith2 객체에만 확장되었음
# #     print("{0} 은(는) 현재 클로킹 상태 입니다.\n".format(wraith2.name))




########################################################################################
#
#   메소드 : 3:53:06
#   class 내의 함수(?)


# 공격유닛
# class AttackUnit:
#     def __init__(self, name, hp, damage):
#         self.name = name
#         self.hp = hp
#         self.damage = damage
#         print("\n{0} 유닛이 생성 되었습니다.".format(self.name))
#         print("체력 : {0}, 공격력 : {1}\n ".format(self.hp, self.damage))

#     # self.~ : 자기자신 클래스내의 ~멤버변수...
#     # name, hp, damage, location : 인자로 받은 변수
#     def attack(self, location):
#         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 : {2}]\n"\
#             .format(self.name, location, self.damage))

#     def damaged(self, damage):
#         self.hp -= damage
#         print("{0} : {1} 데미지를 입었습니다.\n".format(self.name, damage))   
        
#         if self.hp <= 0:
#             print("{0} 유닛이 파괴 되었습니다!\n".format(self.name))
#         else:
#             print("{0}의 현재 체력은 {1} 입니다.\n".format(self.name, self.hp))


# # 파이어뱃 : 공격유닛, 화염방사기.
# firebat1 = AttackUnit("파이어뱃", 50, 16)
# firebat1.attack("5시")

# # 공격을 2번 받는다고 가정
# firebat1.damaged(25)
# firebat1.damaged(25)




########################################################################################
#
#   상속    : 3:59:29
#


# 메딕 : 의무병 -> 공격력이 없음.
# class Unit를 class AttackUnit이 상속받아서 생성


# class Unit:
#     def __init__(self, name, hp):
#         self.name = name
#         self.hp = hp
       

# #class AttackUnit:
# class AttackUnit(Unit):         # Unit class를 상속받는다.
#     def __init__(self, name, hp, damage):
#         Unit.__init__(self, name, hp)   # 초기화, Unit class의 name과 hp를 상속 받는다.
#         #self.name = name   # Unit class에서 상속받아 사용
#         #self.hp = hp       # Unit class에서 상속받아 사용
#         self.damage = damage
#         print("\n{0} 유닛이 생성 되었습니다.".format(self.name))
#         print("체력 : {0}, 공격력 : {1}\n ".format(self.hp, self.damage))

#     # self.~ : 자기자신 클래스내의 ~멤버변수...
#     # name, hp, damage, location : 인자로 받은 변수
#     def attack(self, location):
#         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 : {2}]\n"\
#             .format(self.name, location, self.damage))

#     def damaged(self, damage):
#         self.hp -= damage
#         print("{0} : {1} 데미지를 입었습니다.\n".format(self.name, damage))   
        
#         if self.hp <= 0:
#             print("{0} 유닛이 파괴 되었습니다!\n".format(self.name))
#         else:
#             print("{0}의 현재 체력은 {1} 입니다.\n".format(self.name, self.hp))


# # 파이어뱃 : 공격유닛, 화염방사기.
# firebat1 = AttackUnit("파이어뱃", 50, 16)
# firebat1.attack("5시")

# # 공격을 2번 받는다고 가정
# firebat1.damaged(25)
# firebat1.damaged(25)



########################################################################################
#
#   다중상속    : 4:02:54
#       여러 클래스에서 상속을 받는 것
# 

# class Unit:
#     def __init__(self, name, hp):
#         self.name = name
#         self.hp = hp
       

# #class AttackUnit:
# class AttackUnit(Unit):         # Unit class를 상속받는다.
#     def __init__(self, name, hp, damage):
#         Unit.__init__(self, name, hp)   # 초기화, Unit class의 name과 hp를 상속 받는다.
#         #self.name = name   # Unit class에서 상속받아 사용
#         #self.hp = hp       # Unit class에서 상속받아 사용
#         self.damage = damage
#         print("\n{0} 유닛이 생성 되었습니다.".format(self.name))
#         print("체력 : {0}, 공격력 : {1}\n ".format(self.hp, self.damage))

#     # self.~ : 자기자신 클래스내의 ~멤버변수...
#     # name, hp, damage, location : 인자로 받은 변수
#     def attack(self, location):
#         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 : {2}]\n"\
#             .format(self.name, location, self.damage))

#     def damaged(self, damage):
#         self.hp -= damage
#         print("{0} : {1} 데미지를 입었습니다.\n".format(self.name, damage))   
        
#         if self.hp <= 0:
#             print("{0} 유닛이 파괴 되었습니다!\n".format(self.name))
#         else:
#             print("{0}의 현재 체력은 {1} 입니다.\n".format(self.name, self.hp))

# # 날 수 있는 기능을 가진 클래스
# class Flyable:
#     def __init__(self, flying_speed):
#         self.flying_speed = flying_speed
    
#     def fly(self, name, location):
#         print("{0} : {1} 방향으로 날아 갑니다. [속도 : {2}]\n".format(name, location, self.flying_speed))


# # 공중공격 유닛 클래스
# # 다중상속, AttackUnit와 Flyable에서 모두 받음
# class FlyableAttackUnit(AttackUnit, Flyable):   
#     def __init__(self, name, hp, damage, flying_speed):
#         AttackUnit.__init__(self, name, hp, damage)
#         Flyable.__init__(self, flying_speed)


# # 발키리 : 공중 공격 유닛, 한번에 14발 미사일 발사.
# # 체력 : 200, 공격력 : 6, 비행속도 : 5
# valkyrie = FlyableAttackUnit("발키리", 200, 6, 5)   

# # 날아서 이동
# valkyrie.fly("발키리", "12시")          # 그냥사용
# valkyrie.fly(valkyrie.name, "3시")     # 멤버변수 사용




###################################################################################
#
#   연산자 오버로딩 :   4:10:08
#       자식클래스에서 정의한 메소드를 사용하고 싶은 경우 새롭게 정의 해서 사용할 수 있음.(연산자 오버로딩) 
#


# # 지상유닛에 스피드를 추가하기 위해 인자를 추가
# class Unit:
#     def __init__(self, name, hp, speed):    # 지상유닛의 스피드를 추가하기 위해 speed 추가
#         self.name = name
#         self.hp = hp
#         self.speed = speed
    
#     def move(self, location):
#        print("[지상 유닛 이동]")
#        print("{0} : {1} 방향으로 이동합니다.[속도 : {2}]\n".format(self.name, location, self.speed)) 
       

# #class AttackUnit:
# class AttackUnit(Unit):         # Unit class를 상속받는다.
#     def __init__(self, name, hp, speed, damage):    # Unit에서 추가한 speed 추가
#         Unit.__init__(self, name, hp, speed)   # 초기화, Unit class의 name과 hp, speed를 상속 받는다.
#         self.damage = damage
#         print("\n{0} 유닛이 생성 되었습니다.".format(self.name))
#         print("체력 : {0}, 공격력 : {1}, 속도 : {2}\n ".format(self.hp, self.damage, self.speed))

#     def attack(self, location):
#         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 : {2}]\n"\
#             .format(self.name, location, self.damage))

#     def damaged(self, damage):
#         self.hp -= damage
#         print("{0} : {1} 데미지를 입었습니다.\n".format(self.name, damage))   
        
#         if self.hp <= 0:
#             print("{0} 유닛이 파괴 되었습니다!\n".format(self.name))
#         else:
#             print("{0}의 현재 체력은 {1} 입니다.\n".format(self.name, self.hp))

# # 날 수 있는 기능을 가진 클래스
# class Flyable:
#     def __init__(self, flying_speed): 
#         self.flying_speed = flying_speed
    
#     def fly(self, name, location):
#         print("{0} : {1} 방향으로 날아 갑니다. [속도 : {2}]\n".format(name, location, self.flying_speed))

# # 다중상속, AttackUnit와 Flyable에서 모두 받음
# class FlyableAttackUnit(AttackUnit, Flyable):      
#     def __init__(self, name, hp, damage, flying_speed): 
#         # AttackUnit의 speed(지상 이동속도) 는 공중유닛이므로 0으로 초기화. 
#         AttackUnit.__init__(self, name, hp, 0, damage)
#         Flyable.__init__(self, flying_speed)
    
#     # 연산자 오버로딩하기 위해... move함수를 만듬
#     def move(self, location):
#         print("[공중 유닛 이동]")
#         self.fly(self.name, location)




# # 벌쳐 : 지상 유닛, 기동성이 좋음
# vulture = AttackUnit("벌쳐", 80, 10, 20)

# # 배틀크루저 : 공중유닛, 체력 및 공격력이 좋음
# battlecruiser = FlyableAttackUnit("배틀크루저", 500, 25, 3)

# # 이동
# vulture.move("11시")
# battlecruiser.fly(battlecruiser.name, "9시")

# #
# # 문제점 발생
# #   이동시 지상유닛은 .move 함수를 사용
# #   이동시 공중유닛은 .fly 함수를 사용
# #   따라서 이동을 위해서는 공중유닛인지 지상유닛인지 알아야 함....
# #   문제를 해결하고자 연산자 오버로딩을 통해서 공중유닛, 지상유닛 둘다 .move 함수를 사용하게 변경
# #   Unit(.move) <- AttackUnit : Unit의 move함수는 지상유닛 이동용
# #   Unit(.move) <- AttackUnit <- FlyableAttackUnit(.move() 재정의) -> Flyable 
# #   => 공중유닛 이동은 FlyableAttackUnit에서 재정의된 move() 함수를 통해 이동 

# battlecruiser.move("3시")



##########################################################################################
#
#   pass : 4:17:02
#       일단 통과...암것도 안하고... 일단 pass


# class Unit:
#     def __init__(self, name, hp, speed):    
#         self.name = name
#         self.hp = hp
#         self.speed = speed
    
#     def move(self, location):
#        print("[지상 유닛 이동]")
#        print("{0} : {1} 방향으로 이동합니다.[속도 : {2}]\n".format(self.name, location, self.speed)) 
       

# #class AttackUnit:
# class AttackUnit(Unit):         
#     def __init__(self, name, hp, speed, damage):    
#         Unit.__init__(self, name, hp, speed)  
#         self.damage = damage
#         print("\n{0} 유닛이 생성 되었습니다.".format(self.name))
#         print("체력 : {0}, 공격력 : {1}, 속도 : {2}\n ".format(self.hp, self.damage, self.speed))

#     def attack(self, location):
#         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 : {2}]\n"\
#             .format(self.name, location, self.damage))

#     def damaged(self, damage):
#         self.hp -= damage
#         print("{0} : {1} 데미지를 입었습니다.\n".format(self.name, damage))   
        
#         if self.hp <= 0:
#             print("{0} 유닛이 파괴 되었습니다!\n".format(self.name))
#         else:
#             print("{0}의 현재 체력은 {1} 입니다.\n".format(self.name, self.hp))

# # 날 수 있는 기능을 가진 클래스
# class Flyable:
#     def __init__(self, flying_speed): 
#         self.flying_speed = flying_speed
    
#     def fly(self, name, location):
#         print("{0} : {1} 방향으로 날아 갑니다. [속도 : {2}]\n".format(name, location, self.flying_speed))

# # 다중상속, AttackUnit와 Flyable에서 모두 받음
# class FlyableAttackUnit(AttackUnit, Flyable):      
#     def __init__(self, name, hp, damage, flying_speed): 
#         # AttackUnit의 speed(지상 이동속도) 는 공중유닛이므로 0으로 초기화. 
#         AttackUnit.__init__(self, name, hp, 0, damage)
#         Flyable.__init__(self, flying_speed)
    
#     # 연산자 오버로딩하기 위해... move함수를 만듬
#     def move(self, location):
#         print("[공중 유닛 이동]")
#         self.fly(self.name, location)


# # 건물
# class BuildingUnit(Unit):
#     def __init__(self, name, hp, location):
#         pass


# # 서플라이 디폿 : 건물, 8 유닛 당 1건물 필요
# supply_depot = BuildingUnit("서플라이 디폿", 500, "7시")

# def game_start():
#     print("[알림] 새로운 게임을 시작합니다.")

# def game_over():
#     pass

# game_start()
# game_over()




###########################################################################################
#
#   super() : 4:19:31
#
#       다음 두 문장은 동일한 의미 임    
#       Unit.__init__(self, name, hp, 0)   
#       super().__init__(name, hp, 0)
#       주의사항 
#           super()를 사용시 self는 제외해야 함.
#           다중 상속시 문제 발생

# 건물
# class BuildingUnit1(Unit):
#     def __init__(self, name, hp, location):
#         Unit.__init__(self, name, hp, 0)    # 건물은 이동할 수 없으므로 speed = 0으로 처리
#         #super().__init__(name, hp, 0)
#         self.location = location

# # 문제점 예
# class Unit1:
#     def __init__(self):
#         print("Unit 생성자")

# class Flyable1:
#     def __init__(self):
#         print("Flyable 생성자")

# class FlyableUnit1(Unit1, Flyable1): # Unit1, Flyable1 둘중 먼저 있는 Unit1만 상속 받음
# # class FlyableUnit1(Flyable1, Unit1): # Flyable1, Unit1 둘중 먼저 있는 Flyable만 상속 받음
# # Unit1, Flyable1 둘다 상속 받기 위해서는 따로따로 초기화 해주어야 함.
#     def __init__(self):
#         #super().__init__()
#         Unit1.__init__(self)
#         Flyable1.__init__(self)

# # 드랍쉽
# dropship = FlyableUnit1()   




#########################################################################################################
#
#   스타크래프트 전반전 : 4:23:52
#   스타크래프트 후반전 : 4:33:46
#       starcraft.py 참조
#


#########################################################################################################
#
#   퀴즈 8  : 4:44:43
#
#   주어진 코드를 활용하여 부동산 프로그램을 작성하시오.
#       (출력예제)
#       총 3대의 매물이 있습니다.
#       강남    아파트      매매    10억    2010년
#       마포    오피스텔    전세    5억     2007년
#       송파    빌라        월세    500/50  2000년
#   [code]
#       class House:
#           # 매물 초기화
#           def __init__(self, location, house_type, deal_type, price, completion_year):
#               pass
#
#           # 매물 정보표시
#           def show_detail(self):
#               pass
# 

# class House:    
#     def __init__(self, location, house_type, deal_type, price, completion_year):
#         self.location = location
#         self.house_type = house_type
#         self.deal_type = deal_type
#         self.price = price
#         self.completion_year = completion_year
        

#     def show_details(self):
#         print(self.location,"\t", self.house_type,"\t", self.deal_type,"\t",self.price,"\t", self.completion_year)
#         # print("{0: <12}".format(self.location), end="")
#         # print("{0: <12}".format(self.house_type), end="")
#         # print("{0: <12}".format(self.deal_type), end="")
#         # print("{0: <12}".format(self.price), end="")
#         # print("{0: <12}".format(self.completion_year))

# # Data 입력
# house_list = [] 
# house1 = House("강남", "아파트", "매매", "10억", "2010년")    
# house2 = House("마포", "오피스텔", "전세", "5억", "2007년") 
# house3 = House("송파", "빌라", "월세", "500/50", "2000년")   

# house_list.append(house1)
# house_list.append(house2)
# house_list.append(house3)

# # print("house1 = {0}".format(house1))
# # print("house2 = {0}".format(house2))
# # print("house3 = {0}".format(house3))

# print("총 {0}개의 매물이 있습니다.".format(len(house_list)))
# for house in house_list:
#     house.show_details()



#########################################################################################################
#
#   예외처리 :  4:50:12
#       error가 발생했을때 처리하는 것


# print("나누기 전용 계산기")
# num1 = int(input("첫 번째 숫자를 입력하세요 : "))
# num2 = int(input("두 번째 숫자를 입력하세요 : "))
# print("{0} / {1} = {2}".format(num1, num2, int(num1/num2)))

# 만약 숫자대신 문자를 넣을 경우 Error가 발생하면서 프로그램이 종료됨...
# 이를 방지하기 위해 예외처리를 한다....
# 나누기 전용 계산기
# 첫 번째 숫자를 입력하세요 : 6
# 두 번째 숫자를 입력하세요 : 삼
#   Traceback (most recent call last):
#   File "c:/Python_Test/practice.py", line 1459, in <module>
#   num2 = int(input("두 번째 숫자를 입력하세요 : "))
#   ValueError: invalid literal for int() with base 10: '삼'
# ValueError 발생

# #예외처리
# try:
#     print("나누기 전용 계산기")
#     nums = []
#     nums.append(int(input("첫 번째 숫자를 입력하세요 : ")))
#     nums.append(int(input("두 번째 숫자를 입력하세요 : ")))
#     #nums.append(int(nums[0]/nums[1]))
#     print("{0} / {1} = {2}".format(nums[0], nums[1], nums[2]))

# except ValueError:
#     print("Error! 잘못된 값을 입력하였습니다.")

# except ZeroDivisionError as err:
#     # 0으로 나누었을때 발생하는 오류
#     print("Error! 0으로 나룰 수 없습니다.")
#     print(err)

# except Exception as err:
#     print("Error! 오류발생 : {0}.".format(err))
#     #print(err)




######################################################################################################
#
#   에러 발생시키기 :   4:58:15
#

# try:
#     print("한 자리 숫자 나누기 전용 계산기")
#     num1 = int(input("첫 번재 숫자를 입력하세요 :  "))
#     num2 = int(input("두 번재 숫자를 입력하세요 :  "))
#     if num1 >= 10 or num2 >= 10:
#         raise ValueError    # ValueError 발생시킴...

#     print("{0} / {1} = {2}".format(num1, num2, int(num1/num2)))

# except ValueError:
#     print("잘못된 값을 입력하였습니다. 한 자리 숫자만 입력하세요!")



#######################################################################################################
#
#   사용자 정의 예외처리 :  05:01:06
#

# class BigNumberError(Exception):
#     def __init__(self, msg):
#         self.msg = msg
    
#     def __str__(self):
#         return self.msg

# try:
#     print("한 자리 숫자 나누기 전용 계산기")
#     num1 = int(input("첫 번재 숫자를 입력하세요 :  "))
#     num2 = int(input("두 번재 숫자를 입력하세요 :  "))
#     if num1 >= 10 or num2 >= 10:
#         raise BigNumberError("클래스에서 정의된 Error MSG, 입력값 : {0}, {1}".format(num1, num2))    

#     print("{0} / {1} = {2}".format(num1, num2, int(num1/num2)))

# except ValueError:
#     print("잘못된 값을 입력하였습니다. 한 자리 숫자만 입력하세요!")

# except BigNumberError as err:
#     print("한자리 숫자만 입력하라고....")
#     print(err)



#######################################################################################################
#
#   Finally :  05:04:26
#       예외처리에서 무조건 실행되는 문장... 에러가 발생하던 하지않던 무조건 실행
#
# class BigNumberError(Exception):
#     def __init__(self, msg):
#         self.msg = msg
    
#     def __str__(self):
#         return self.msg

# try:
#     print("한 자리 숫자 나누기 전용 계산기")
#     num1 = int(input("첫 번재 숫자를 입력하세요 :  "))
#     num2 = int(input("두 번재 숫자를 입력하세요 :  "))
#     if num1 >= 10 or num2 >= 10:
#         raise BigNumberError("클래스에서 정의된 Error MSG, 입력값 : {0}, {1}".format(num1, num2))    

#     print("{0} / {1} = {2}".format(num1, num2, int(num1/num2)))

# except ValueError:
#     print("잘못된 값을 입력하였습니다. 한 자리 숫자만 입력하세요!")

# except BigNumberError as err:
#     print("한자리 숫자만 입력하라고....")
#     print(err)

# finally:
#     print("finally 문장 : 이용해 주셔서 감사합니다.")



#######################################################################################################
#
#   퀴즈9 :  05:06:18
#
# 동네에 항상 대기 손님이 있는 치킨집이 있습니다.
# 대기 손님의 치킨 요리 시간을 줄이고자 자동 주문 시스템을 제작하였습니다.
# 시스템 코드를 확인하고 적절한 예외처리 구문을 넣으시오.

# 조건 1 : 1보다 작거나 숫자가 아닌 값이 들어올 때는 ValueError로 처리
#         출력메세지 : "잘못된 값을 입력 하였습니다."
# 조건 2 : 대기 손님이 주문할 수 있는 치킨의 수량은 10마리로 한정
#         치킨 소진 시 사용자 정의  에러 [SoldOutError]를 발생시키고 프로그램 종료
#         출력메세지 : "재고가 소진되어 더 이상 주문을 받기 않습니다."
# [Code]


# class SoldOutError(Exception):
#     def __init__(self, msg):
#         self.msg = msg
    
#     def __str__(self):
#         return self.msg




# chicken = 10
# waiting = 1     # 홀 안에는 현재 만석. 대기번호 1번부터 시작
# maxOrder = 10

# while(True):
#     print("\n[남은 치킨 : {0} 마리]".format(chicken))
#     order = int(input("치킨 몇 마리 주문하시겠습니까?"))
#     try:
#         if order < 1 :
#             raise ValueError
#         elif order > maxOrder:
#             print("최대 주문수량은 {0} 마리 입니다.".format(maxOrder))
#         elif order > chicken:     # 남은 수량보다 주문량이 많을 경우
#             print("재료가 부족합니다. 현재 남은 수량은 {0} 마리 입니다".format(chicken))
#         else :
#             print("[대기번호 : {0}] {1} 마리 주문이 완료 되었습니다.".format(waiting, order))
#             waiting += 1
#             chicken -= order
        
#         if chicken == 0 : 
#             raise SoldOutError("재고가 소진되어 더 이상 주문을 받기 않습니다.")        
         
#     except ValueError:  # 문자인 경우 자동으로  ValueError 발생
#         print("Error : 잘못된 값을 입력하셨습니다. 입력값 : {0}".format(order))
    
#     except SoldOutError as err:
#         print(err)   
#         break



#######################################################################################################
#
#   모듈 :  05:14:25
#       필요한 것들을 묶음으로 모아둔 파일?
#       모듈(module)은 각종 변수, 함수, 클래스를 담고 있는 파일

# import theater_module
# theater_module.price(3)
# theater_module.price_morning(4)
# theater_module.price_soldier(5)

# import theater_module as mv
# mv.price(4)
# mv.price_morning(5)
# mv.price_soldier(6)

# from theater_module import *
# price(3)
# price_morning(4)
# price_soldier(5)

# from theater_module import price, price_morning
# price(5)
# price_morning(6)

# from theater_module import price_soldier as price
# price(5)



#######################################################################################################
#
#   패키지 :  05:24:10
#       모듈을 모아놓은 집합, 쉽게 하나의 디렉토리에 모듈들을 모아놓은 것?
#       패키지(package)는 여러 모듈을 묶은 것
#       travel 폴더(패키지) - tailand.py, vietnam.py, __init__.py 

# #import travel.thailand.ThailandPackage  # 클래스, 함수를 직접 임포트할 수 없음. 사용불가, Error 발생. from ~ import에서는 사용가능
# import travel.thailand  # 패키지 or 모듈을 임포트 한 후, 클래스, 함수를 사용할 수 있음
# trip_to = travel.thailand.ThailandPackage()
# trip_to.detail()

# from travel.thailand import ThailandPackage
# trip_to = ThailandPackage()
# trip_to.detail()

# from travel import vietnam
# trip_to = vietnam.VietnamPackage()
# trip_to.detail()



#######################################################################################################
#
#   __all__ :  05:30:29
#       
    
#from random import *
#from travel import *   

# 아래 두 문장 실행시 Error 발생
# * 을 사용한다는 것은 모든 것을 사용하는데, 개발자가 공개범위를 설정해야 함.
# trip_to = vietnam.VietnamPackage()
# trip_to.detail()

# __init__.py 파일
# __all__ = ["vietnam"] 추가 후, 아래문장은 정상적으로 실행 됨.
# trip_to = vietnam.VietnamPackage()
# trip_to.detail()

# # __init__.py 파일
# # __all__ = ["vietnam", "thailand"] 추가 후, 아래문장은 정상적으로 실행 됨.
# trip_to = thailand.ThailandPackage()
# trip_to.detail()



#######################################################################################################
#
#   모듈 직접 실행 :  05:34:15
#   thailand.py파일을 직접 실행할 수 도 있음   
#

# from travel import *
# trip_to = thailand.ThailandPackage()
# trip_to.detail()



#######################################################################################################
#
#   패키지, 모듈 위치 :  05:37:00
#       import inspect    
#
 
# import inspect
# import random

# # random 모듈이 어느위치에 있는지 확인하는 방법 
# print(inspect.getfile(random))
# print(inspect.getfile(thailand))




#######################################################################################################
#
#   pip install :  05:40:32
#       pip를 이용한 package 설치 
#       pypi  검색
#       pip list : 현재 설치된 패키지 검색
#       pip show beautifulsoup4 : beautifulsoup4의 정보를 표시 
#       pip install --upgrade beautifulsoup4 : beautifulsoup4을 upgrade 
#       pip install beautifulsoup4  : 설치
#       pip uninstall beautifulsoup4 : 삭제

# from bs4 import BeautifulSoup
# soup = BeautifulSoup("<p>Some<b>bad<i>HTML")
# print(soup.prettify())



#######################################################################################################
#
#   내장함수 :  05:46:04
#       input : 사용자 입력을 받는 함수   
#       google 검색 : list of python builtins     
#      

# dir : 어떤 객체를 넘겨줬을 때 그 객체가 어떤 변수와 함수를 가지고 있는지 표시
# print(dir())
# print("-----------------------------------------------------------------")
# import random       # 외장함수
# print(dir())
# print("-----------------------------------------------------------------")
# import pickle
# print("-----------------------------------------------------------------")
# print(dir())
# print("-----------------------------------------------------------------")
# print(dir(random))
# print("-----------------------------------------------------------------")
# lst = [1,2,3]
# print(dir(list))
# print("-----------------------------------------------------------------")


#######################################################################################################
#
#   외장함수 :  05:50:38
#       google 검색 : list of python modules      
#      

# glob : 경로 내의 폴더 / 파일 목록 조회(윈도우 dir)
# import glob
# print(glob.glob("*.py"))


# os : 운영체제에서 제공하는 기본 기능(폴더 생성 , 삭제 등)
#import os
# print(os.getcwd())  # 현재 디렉토리 표시

# foler = "sample_dir"

# if os.path.exists(folder):      # sample_dir 폴더가 있으면
#     print("이미 존재하는 폴더 입니다")
#     os.rmdir(folder)            # 폴더 삭제
#     print(folder, "폴더를 삭제하였습니다")
# else:
#     os.makedirs(folder)     # 폴더 생성
#     print("{0} 폴더를 생성하였습니다.".format(folder))


# time : 시간관련 함수
# import time
# print(time.localtime())
# print(time.strftime("%Y-%m-%d %H:%M:%S"))

# # datetime
# import datetime
# print("오늘 날짜는 ", datetime.date.today())

# # timedelta : 두 날짜 사이의 간격
# today = datetime.date.today()   # 오늘 날짜 저장
# td = datetime.timedelta(days=100)
# print("오늘 부터 100일 후는", today+td)

# d = datetime.date(2009, 10, 13)

# print("오늘은 {0}일 되는 날 입니다.".format(today - d))



#######################################################################################################
#
#   퀴즈 10 :  05:58:49
#      

# Quiz) 프로젝트 내에 나만의 시그니처를 남기는 모듈을 만드시오.

# 조건 : 모듈 파일명은 byme.py로 작성

# (모듈 사용 예제)
# import byme
# byme.sign()

# (출력 예제)
# 이 프로그램은 나도코딩에 의해 만들어졌습니다.
# 유튜브 : http://youtube
# 이메일 : nadocoding@gmail.com

import byme
byme.sign()
