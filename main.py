import numpy as np 
import theano 
import theano.tensor as T 
import lasagne
import lasagne.layers as layers
import model
import load
import time
import matplotlib.pyplot as plt


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


print("Loading data...")
X_train, y_train, X_val, y_val = load.loadDataset()

input_var = T.tensor4('inputs')
target_var = T.tensor4('target')

print("Building model and compiling functions...")

batchsize = 128

# Network
#network = Model.OneLayerMLP(batchsize, input_var)
network = model.simpleConv(input_var)

# Loss Function
prediction = lasagne.layers.get_output(network)
loss = T.mean(lasagne.objectives.squared_error(prediction, target_var))

# Optimizer
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

# Loss Function for validation
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = T.mean(lasagne.objectives.squared_error(prediction, target_var))

# Compiling 
train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], test_loss)
gen_fn = theano.function([input_var], test_prediction)

num_epochs = 100

train_loss_list = []
val_loss_list = []
epoch_list = []

min_loss = 100
p=0
print("Starting training...")
for epoch in range(num_epochs):
	train_err = 0
	train_batches = 0
	start_time = time.time()
	for batch in iterate_minibatches(X_train, y_train, batchsize):
		inputs, targets = batch
		train_err += train_fn(inputs, targets)
		train_batches += 1

	val_err = 0
	val_batches = 0
	for batch in iterate_minibatches(X_val, y_val, batchsize):
		inputs, targets = batch
		err = val_fn(inputs, targets)
		val_err += err
		val_batches += 1

	train_loss_list.append(train_err / train_batches)
	val_loss_list.append(val_err / val_batches)
	epoch_list.append(epoch)

	print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
	print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

	if min_loss > (val_err / val_batches) :
		p=0
	else:
		p += 1 
	if p>=10:
		break

plt.plot(train_loss_list)
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.show()

plt.plot(val_loss_list)
plt.ylabel('Validation Loss')
plt.xlabel('Epochs')
plt.show()

img_gen = X_val[0:128,:,:,:]
print img_gen.shape

result_gen = gen_fn(img_gen)
print result_gen.shape

full_images = load.rebuildImage(img_gen, result_gen)
load.saveImages(full_images)











