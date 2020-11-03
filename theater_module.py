# 일반가격(10,000원)
def price(people):
    print("{0}명 가격은 {1}원 입니다.".format(people, people * 10000))

# 조조할인 가격(6,000원)
def price_morning(people):
    print("조조할인 {0}명 가격은 {1}원 입니다.".format(people, people * 6000))
    
# 군인 할인 가격(4,000원)
def price_soldier(people):
    print("군인 {0}명 가격은 {1}원 입니다.".format(people, people * 4000))