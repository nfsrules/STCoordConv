import torch
import numpy as np
from IPython import display
from torch.autograd import Variable
import matplotlib.pyplot as plt
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from dataset import get_speed, stack_speed


def to_rgb_img(tensor):
	'''Convert 3-channel input tensor to RGB image
	
	'''    
	return np.moveaxis(tensor*255, 0, 2)


def get_coordinates(onehot, canvas_size=64):
	position = onehot.argmax(1)
	onehot[0, position] = 255
	im = onehot.reshape(-1,canvas_size,canvas_size)
	pos = np.where(im == 255)
	
	return pos[1], pos[2]


def paint_prediction(im, square_size=4, c=[10,10]):
	cx = int(c[0])
	cy = int(c[1])
	im[
		cy - square_size//2:cy + square_size//2, 
		cx - square_size//2:cx + square_size//2
		] = 150 
	
	return im


def train_model(epochs, net, criterion, optimizer, trainloader):
	iters = 0
	net.train()

	for epoch in range(1, epochs + 1):
		iters = 0
		for batch_idx, (data, target) in enumerate(trainloader):
			data, target = Variable(data), Variable(target)
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = net(data)
			loss = criterion(output, target.float())
			loss.backward()
			optimizer.step()
			iters += len(data)
			print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
					epoch, iters, len(trainloader.dataset),
					100. * (batch_idx + 1) / len(trainloader), loss.data.item()), end='\r', flush=True)
		print("")


def eval_model(net, criterion, testloader):
	net.eval()
	total_loss = 0
	index = 0
	for batch_idx, (data, target) in enumerate(testloader):
		data, target = Variable(data), Variable(target)
		data, target = data.to(device), target.to(device)
		output = net(data)
		loss = criterion(output, target.float())
		total_loss = total_loss + loss
		index = index + 1
	
	#loss = total_loss / (batch_idx + 1)  # average loss per batch
	loss = total_loss / (index + 1)  # average loss per batch
	loss = loss / data.shape[0]  # average loss per sample
	print("Average loss per sample =", loss)


def play_raw_data(ds, test_idx, lim=100, canvas_size=64):
	"""Plat sucession of frames from input data. Annotate CAN readings.
	
	data: CAN_dataframe
	f_init: initial frame
	f_end: end frame
	"""
	# Get input shape
	c, img, y, s = ds[0]
	shape = img.shape
	nrb_channel = shape[1]

	fig, ax = plt.subplots(1)
	for i, index in enumerate(test_idx): #range(f_init,f_end,skip):
		c, img, y, s = ds[index]
		display.clear_output(wait=True)
		display.display(plt.gcf())

		if nrb_channel == 1:
			plt.imshow(img.numpy()[0,:,:])
		else:
			plt.imshow(to_rgb_img(img.numpy()))

		plt.axis('off')
		plt.title('speed {}; angle{}'.format(round(s[0],3), round(s[1],3)))
		if i > lim:
			break
	plt.close()
	

def play_predictions_openloop(ds, test_idx, model, lim=100, canvas_size=64):
	"""Plat sucession of frames from input data. Annotate CAN readings.
	
	data: CAN_dataframe
	f_init: initial frame
	f_end: end frame
	"""
	model.eval
	fig, ax = plt.subplots(1)
	for i, index in enumerate(test_idx): #range(f_init,f_end,skip):
		c, img, y, s = ds[index]
		
		# Forward pass
		output = model(img.unsqueeze(0))
		#position = output.argmax(1)
		#output[0, position] = 255
		img = output.reshape(-1,canvas_size,canvas_size)
		display.clear_output(wait=True)
		display.display(plt.gcf())
		plt.imshow(img.detach().cpu().numpy()[0,:,:], cmap='inferno')
		plt.axis('off')
		if i > lim:
			break
			
	plt.close()
	

def play_predictions_closedloop1D(ds, seed, model, lim=100, canvas_size=64):
	"""Plat sucession of frames from input data. Annotate CAN readings.
	
	data: CAN_dataframe
	f_init: initial frame
	f_end: end frame
	"""
	model.eval
	fig, ax = plt.subplots(1)
	i = 0
	# Get seed image from dataset
	c, img, y, s = ds[seed]
	
	while i < lim:
		# Forward pass
		output = model(img.unsqueeze(1))
		
		# Convert output onehot to image
		position = output.argmax(1)
		output[0, position] = 255
		img = output.reshape(-1,canvas_size,canvas_size)
		
		# Display image
		display.clear_output(wait=True)
		display.display(plt.gcf())
		plt.imshow(img.detach().cpu().numpy()[0,:,:], cmap='inferno')
		plt.axis('off')
		
		# Push output image to input on closed loop
		img = output.reshape(-1,canvas_size,canvas_size)/255
		i = i + 1  # Control index
			
	plt.close()


def play_predictions_closedloop3D(ds, seed, model, lim=10, canvas_size=28):
    model.eval
    fig, ax = plt.subplots(1)
    i = 0
    # Get seed image from dataset
    current_center, img, y, s = ds[seed]

    while i < lim:
        # Forward pass current_image
        output = model(img.unsqueeze(0))
        pred_y = output.reshape(-1,canvas_size,canvas_size)

        # Convert output onehot to image
        position = output.argmax(1)
        output[0, position] = 255
        
        # Get predicted center
        pred_center = np.where(pred_y == 255)
        pred_center = int(pred_center[1]), int(pred_center[2])

        # Display image
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.imshow(pred_y.detach().cpu().numpy()[0,:,:], cmap='inferno')
        plt.axis('off')

        # Get speed
        speed, angle = get_speed(pred_center, current_center)
        # Stack speed
        img = stack_speed(speed, angle, pred_y, pred_center)
        
        # Update past center
        current_center = pred_center
        i = i + 1  # Control index

    plt.close()