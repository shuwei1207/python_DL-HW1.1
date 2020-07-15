# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:55:28 2020

@author: SeasonTaiInOTA
"""
import matplotlib.pyplot as plt
import numpy as np

print('-----------loading data-----------')
#資料載入
train_data = np.load('train.npz') 
test_data = np.load('test.npz')

print('-----------data loaded-----------')

#輸出種類
number = 10

print('-----------processing data-----------')

#圖形整理
x_train = train_data['image'].reshape(12000, 784)
y_train = np.array([[0]*number]*12000)

#01轉換
for i in range(12000):
    y_train[i][int(train_data['label'][i])] = 1 

#圖形整理
x_test = test_data['image'].reshape(5768, 784)
y_test = np.array([[0]*number]*5768)

#01轉換
for i in range(5768):
    y_test[i][int(test_data['label'][i])] = 1

print('-----------data processed-----------')


class DNN():
    
    #係數初始化 problem2 可以調weight
    def __init__(self, input_size , hidden_size , output_size , weight_init= 0.01):
        self.params = {}
        self.params['W1'] = weight_init * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
       
    #預測出的 y
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        #layer之間的方法
        a1 = np.dot(x, W1) + b1
        z1 = self.relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.relu(a2)
        return y
    
    #係數更新
    def gradient(self, x, t):
        #先取得上一輪的值
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grad = {}
        
        #直接一次訓練完成 forward更新
        batch_num = 12000
        a1 = np.dot(x, W1) + b1
        z1 = self.relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.relu(a2)
        
        dy = (y - t) / batch_num
        
        #參數backward更新
        grad['W2'] = np.dot(z1.T, dy)
        grad['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = self.sigmoid(a1) * dz1
        grad['W1'] = np.dot(x.T, da1)
        grad['b1'] = np.sum(da1, axis=0)
        #print(grad)
        return grad
    
    #loss function
    def loss(self, x, r):
        y = self.predict(x)
        return self.cross_entropy(y, r)
    
    #loss function的運算
    def cross_entropy(self, y , r):
        delta = 1e-8
        return -np.sum(r * np.log (y+delta) )
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
    
    def relu(self, x):
        return np.maximum(0, x)
    
    #error rate
    def error(self, x, r):
        y = self.predict(x)
        y = np.argmax(y, axis=1) #轉換種類
        r = np.argmax(r, axis=1) #轉換種類
        err = np.sum(y != r) / float(x.shape[0])
        return err
    
    

print('-----------start training-----------')
network = DNN(input_size=784, hidden_size=10 , output_size=10)

#參數設定
iters_num = 1000
learning_rate = 0.001

#problem 1
train_loss_list = []
train_err_list = []
test_err_list = []

for i in range(iters_num):
    
    #取得更新
    grad = network.gradient(x_train, y_train)
    
    #print(network.params)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    #train loss function
    loss = network.loss(x_train, y_train)
    train_loss_list.append(loss)
    
    #error list
    train_err = network.error(x_train, y_train)
    test_err = network.error(x_test, y_test)
    train_err_list.append(train_err)
    test_err_list.append(test_err)
    
    print(i,'epoch: ','train err=', train_err ,'test err=', test_err)
    
    #latent distribution
    if i ==20:
        y_20 = network.predict(x_test)
        y_20 = np.argmax(y_20, axis=1)
    
    if i ==80:
        y_80 = network.predict(x_test)
        y_80 = np.argmax(y_80, axis=1)
    

print('-----------end training-----------')

plt.plot(train_loss_list)
plt.show()
plt.plot(train_err_list)
plt.show()
plt.plot(test_err_list)
plt.show()

#confusion matrix
y = network.predict(x_test)
y = np.argmax(y, axis=1)
r = np.argmax(y_test, axis=1)