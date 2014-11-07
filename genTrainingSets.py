import random

#numbers = [0,1,2,3,4,5,6,7,8,9]
numbers = [0,1]

numlist = []

def genList(n,k):
    for i in range(n):
        tmp = []
        tmp2 = []
        fsum = []
        sum = 0
        for j in range(k):
            num = random.choice(numbers)
            tmp.append(num)
            sum = sum + num
        if sum > 4:
            fsum.append(1)
        else:
            fsum.append(0)
        tmp2.append(tmp)
        tmp2.append(fsum)
        numlist.append(tmp2)

def main():
    # genList(Number of Training Sets , Size of the set)
    genList(1,10)
    print(numlist)
             
if __name__=="__main__":
    main()





