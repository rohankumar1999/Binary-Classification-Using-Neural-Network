import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
file=open('housepricedata.csv','rt')
fr=file.readlines()
file.close()
ip=list()

for line in fr:
	ip.append(line.strip("\n").split(","))
random.shuffle(ip)


x=list()
y=list()
# iters=100

train_set=ip[:math.floor(0.8*len(ip))]
test_set=ip[math.floor(0.8*len(ip)):]
for item in train_set:
	y.append(list((item[len(item)-1])))
	item.pop()
	x.append(item)
x=np.array(x)
y=np.array(y)
x=x.astype(np.float)
y=y.astype(np.float)
x=(x-np.mean(x,axis=0))/(np.var(x,axis=0))
# y=(y-np.mean(y,axis=0))/(np.var(y,axis=0))
# print(y)
class NeuralNets:
	def __init__(self,inputs,target,neurons):
		
		self.input= inputs
		self.target=target
		self.input_shape=self.input.shape[1]
		self.target_shape=self.target.shape[1]
		self.neurons=neurons
		self.weights=list()

		w=2 * random.random((self.input_shape, self.neurons[0])) - 1
		self.weights.append(w)
		for i in range(1,self.neurons.shape[0]):
			wt=2 * random.random((self.neurons[i-1], self.neurons[i])) - 1
			self.weights.append(wt)
		self.target_weights=2 * random.random((self.neurons[len(neurons)-1], self.target_shape)) - 1
	def train(self,iters,lr):
		# print(self.weights)
		x_axis=list()
		y_axis=list()
		for i in range(iters):
			layers=[None]*len(self.neurons)
			layers[0]=self.activation(np.dot(self.input,self.weights[0]))
			for j in range(1,self.neurons.shape[0]):
				layers[j]=self.activation(np.dot(layers[j-1],self.weights[j]))
			output=self.activation(np.dot(layers[len(self.neurons)-1],self.target_weights))	
			# print(len(output),len(self.target))
			# print(self.loss(output))
			# print(i)
			x_axis.append(i)
			y_axis.append(self.loss(output))
			error=self.target-output
			delta=error*self.derivative(output)
			self.target_weights+= lr*np.dot(layers[len(layers)-1].T,delta)
			delta=np.dot(delta,self.target_weights.T)*self.derivative(layers[len(layers)-1])
			self.weights[len(self.weights)-1]+=lr*np.dot(layers[len(layers)-2].T,delta)
			for i in reversed(range(1,len(self.neurons)-1)):
				delta=np.dot(delta,self.weights[i+1].T)*self.derivative(layers[i])
				self.weights[i]+=lr*np.dot(layers[i-1].T,delta)
			delta=np.dot(delta,self.weights[1].T)*self.derivative(layers[1])
			self.weights[0]+=lr*np.dot(self.input.T,delta)
		# print(self.weights)
		# print(x_axis)
		plt.plot(x_axis,y_axis)
		plt.show()
	def activation(self,z):
		return 1/(1+np.exp(-z))
	def derivative(self,z):
		return z*(1-z)
	def loss(self,y_pred):
		return np.square(np.subtract(y_pred,self.target)).mean()
	def predict(self,x,y):
		self.c=0
		self.total=0
		tp=fp=tn=fn=0
		layers=[None]*len(self.neurons)
		layers[0]=self.activation(np.dot(x,self.weights[0]))
		for i in range(1,self.neurons.shape[0]):
			layers[i]=self.activation(np.dot(layers[i-1],self.weights[i]))
		overall=self.activation(np.dot(layers[len(self.neurons)-1],self.target_weights))
		# print(overall)
		for j in range(len(x)):

			self.total+=1
			layers=[None]*len(self.neurons)
			layers[0]=self.activation(np.dot(x[j],self.weights[0]))
			for i in range(1,self.neurons.shape[0]):
				layers[i]=self.activation(np.dot(layers[i-1],self.weights[i]))
			output=self.activation(np.dot(layers[len(self.neurons)-1],self.target_weights))	
			pred=0.

			if output>=np.mean(overall):
				pred=1.
			if pred==y[j] and pred==1:
				
				tp+=1
			elif pred==1 and y[j]==0:
				fp+=1
			elif pred==0 and y[j]==1:
				fn+=1
			elif pred==0 and y[j]==0:
				tn+=1
				
			# 	print('yes',output,y[j])
			# else:
			# 	print('no',output,y[j])
		# print('accuracy: ',self.c/self.total,self.c,self.total)
		precision=(tp/(tp+fn))
		recall=(tp/(tp+fp))
		print('precision: ',precision,'recall: ',recall)
		f1_score=(2*precision*recall)/(precision+recall)
		print('accuracy: ',(tp+tn)/(tp+tn+fp+fn))
	# return f1_score,(tp+tn)/(tp+tn+fp+fn)
		print('f1_score: ',f1_score)
neurons=[5,5]


if len(neurons)==1:
	neurons.append(1)
neurons=np.array(neurons)

nn=NeuralNets(x,y,neurons)
nn.train(500,0.005)
test_x=list()
test_y=list()
for item in test_set:
	test_y.append(list((item[10])))
	item.pop()
	test_x.append(item)
test_x=np.array(test_x)
test_y=np.array(test_y)
test_x=test_x.astype(np.float)
test_y=test_y.astype(np.float)
test_x=(test_x-np.mean(test_x,axis=0))/(np.var(test_x,axis=0))

nn.predict(test_x,test_y)