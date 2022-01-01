import numpy

def mean_square_error_loss(network_target, network_predicted):

	gradient = network_target - network_predicted
	loss_metric = gradient ** 2

	return gradient, loss_metric

def cross_entropy_loss(network_target, network_predicted):

	num_classes = network_target.shape[0]
	
	target_label = network_target.argmax()

	loss_metric = -numpy.log(network_predicted[target_label])

	gradient = numpy.zeros(num_classes)
	gradient[target_label] = -1 / network_predicted[target_label]

	return gradient, loss_metric

def loss_from_function(loss, network_target, network_predicted):

    if loss == 'mse':
        return mean_square_error_loss(network_target, network_predicted)
    if loss == 'categorical_crossentropy':
        return cross_entropy_loss(network_target, network_predicted)

