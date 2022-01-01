import numpy
import scipy.special

def sigmoid(x):
	return scipy.special.expit(x)
 
def sigmoid_gradient(x):
	return x * (1.0 - x)

def relu(x):
	return numpy.maximum(0,x)
 
def relu_gradient(x):
	return (x > 0) * 1

def tanh(x):
	return numpy.tanh(x)
 
def tanh_gradient(x):
	return 1 - x ** 2

def softmax(x):
	exps = numpy.exp(x - x.max())
	return exps / numpy.sum(exps)

# this is wrong!
def softmax_gradient(x):
	dx_ds = numpy.diag(x) - numpy.dot(x, x.T)
	return dx_ds.sum(axis=0).reshape(-1, 1) 

def lambda_from_function(activaiton_name):

	if activaiton_name == 'sigmoid':
		return lambda x: sigmoid(x), lambda x: sigmoid_gradient(x)
	elif activaiton_name == 'relu':
		return lambda x: relu(x), lambda x: relu_gradient(x)
	elif activaiton_name == 'tanh':
		return lambda x: tanh(x), lambda x: tanh_gradient(x)
	elif activaiton_name == 'softmax':
		return lambda x: softmax(x), lambda x: softmax_gradient(x)