#! /usr/bin/python
# BackPropagation Network on a binary training sequence  
# Neural Networks Simulation 
# Author : Snehesh
# Mail   : mail@snehesh.me
# 
# Note
#           To train the network with 2500 data sets would take around 45mins or so , So to 
# speed up the process we can use pypy instead of python
#
import math
import random
import string

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix
def Matrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh 
def sigmoid(x):
    return math.tanh(x)

# derivative of sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = Matrix(self.ni, self.nh)
        self.wo = Matrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = Matrix(self.ni, self.nh)
        self.co = Matrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=2500, alpha=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, alpha, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


# Generating Sample Learning Data

def genList(n,k):
    print('Generating Training Sets ')
    numbers = [0,1]
    numlist = []
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
    return numlist

def demoTest():
    # Assign a training set of 2500 sets of 10 inputs each
    print('Intiating training  ...')
    train = genList(2500,10)
    test = []
    n = NN(10,10,1)
    n.train(train)
    print('Completed training , lets start testing ...')
    while(1):
        try:
            print('Input a test pattern ...')
            n.test(input('> '))
        except KeyboardInterrupt:
            print('Shuting down the Neural Net')
            break
    print('Testing Back Propagation Neural Net was Successful')



if __name__ == '__main__':
    demoTest()
