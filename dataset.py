import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import cv2
from itertools import chain


def draw_box(canvas, center, size, value=1):
    canvas[
        0,
        center[0] - size//2:center[0] + size//2, 
        center[1] - size//2:center[1] + size//2
        ] = value

    return canvas


def xy_transform(dp):
    '''Build X,Y pairs from Dataset class.
    X: 3 channel tensor
    Y: Center of the box
    '''
    c, x, y, s = dp
    return x, y[0,:]


class TransformedDataset(Dataset):
    
    def __init__(self, ds, xy_transform):
        self.ds = ds
        self.xy_transform = xy_transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.xy_transform(self.ds[index])


def get_quadrant_indexes(ds):
    '''Get indexes on train and test partitions that belongs
    to different quadrants
    
    '''
    train_quadrant_indices = []
    test_quadrant_indices = []

    for i, (c, p, im, s) in enumerate(ds):
        if c[0] > ds.canvas_size * 0.5 and \
            c[1] > ds.canvas_size * 0.5:
            test_quadrant_indices.append(i)
        else:
            train_quadrant_indices.append(i)
            
    return torch.tensor(train_quadrant_indices), torch.tensor(test_quadrant_indices)

 
def get_speed(current_center, past_center):
    '''Estimate actor speed and angle based on previous/current egopose
    Returns:
        speed: m/s
        angle: degrees
        
    '''
    # Get past center
    past_cx, past_cy = past_center
    # Get speed vector as displacement (assuming fix sampling freq @ 1 pixel/m)
    current_cx, current_cy = current_center
    speedx, speedy = (current_cx-past_cx, current_cy-past_cy)
    # Convert speed vector to polar coordinates
    speed, speed_angle = cv2.cartToPolar(speedx, speedy, angleInDegrees=True)
    
    return speed[0][0], speed_angle[0][0]


def stack_speed(speed, angle, onehot, center, canvas_size=28):
    # Normalize onehot
    onehot = onehot/255
    speed = (speed-0)/(1-0)  # From 0 - 1 
    # Normalize angle from -1 to 1
    #angle = 2*((angle-0)/(360-0)) - 1  ## Could be a good option
    angle = (angle-0)/(360-0)  # From 0 - 1 
    # Create three channel image
    ch1 = torch.zeros((1, canvas_size, canvas_size))
    ch2 = torch.zeros((1, canvas_size, canvas_size))

    # Paint ch1 with speed module
    ch1[0, center[0], center[1]] = speed
    ch2[0, center[0], center[1]] = angle

    # Stack results on a 3D-Tensor
    onehot = torch.stack((onehot, ch1.to('cuda'), ch2.to('cuda')), 1)
    onehot = onehot[0,:,:,:]
    return onehot


class Snake(Dataset):
    
    def __init__(self, canvas_size=64, square_size=9, speed_channel=False, future=True, square=False, complexity=False):
        self.canvas_size = canvas_size
        self.conv_kernel = (1, 1, square_size, square_size)
        self.square_size =  square_size
        self.speed_channel = speed_channel
        self.future = future
        self.square = square
        self.complexity = complexity

        if self.complexity:  # Snake moves forward and backward
            self.centers =   [
                        (i, j) for i in range(0, self.canvas_size -1) 
    
                        for j in chain(range(0, self.canvas_size-1), 
                                       range(self.canvas_size-1, 0, -1))
                        ]
            print('High complexity: Snake moves forward and backward at fixed speed.')

        else:  # Snake moves only forward
            self.centers = [
                        (i, j) for i in range(0, self.canvas_size - 1)
                               for j in range(0, self.canvas_size - 1)
                           ] 
            print('Low complexity: Snake moves only forward at fixed speed.')

        self.len = len(self.centers)

        print('Dataset generated... {} available instances'.format(self.len))
        
    def __getitem__(self, index):
        # Current center
        center = self.centers[index]
        # X: empty canvas with a point on it (current center)
        onehot = torch.zeros((1, self.canvas_size, self.canvas_size))
        
        # Paint square
        if self.square:
            onehot = draw_box(onehot, center, self.square_size)
     
        else:
            onehot[0, center[0], center[1]] = 1

        if self.speed_channel:
            # Compute speed using the centers
            if index == 0: # Speed CAN NOT BE ESTIMATED
                # First frame, set speed to zero
                speed, angle = 0, 0
                # Set speed channels to 0
                ch1 = torch.zeros((1, self.canvas_size, self.canvas_size))
                ch2 = torch.zeros((1, self.canvas_size, self.canvas_size))
            else:
                # Other frames, get the previous center
                prev_center = self.centers[index-1]
                # Calculate speed
                speed, angle = get_speed(center, prev_center)
                # Normalize speed module from 0 to 1
                ### For the moment, the max speed is 1 m/sec
                speed = (speed-0)/(1-0)  # From 0 - 1 
                # Normalize angle from -1 to 1
                #angle = 2*((angle-0)/(360-0)) - 1  ## Could be a good option
                angle = (angle-0)/(360-0)  # From 0 - 1 
                # Create three channel image
                ch1 = torch.zeros((1, self.canvas_size, self.canvas_size))
                ch2 = torch.zeros((1, self.canvas_size, self.canvas_size))
                
                if self.square:
                    ch1 = draw_box(ch1, center, self.square_size, speed)
                    ch2 = draw_box(ch1, center, self.square_size, angle)
                else:
                    # Paint ch1 with speed module
                    ch1[0, center[0], center[1]] = speed
                    ch2[0, center[0], center[1]] = angle

            # Stack results on a 3D-Tensor
            onehot = torch.stack((onehot, ch1, ch2), 1)
            onehot = onehot[0,:,:,:]
        else:
            speed, angle = (0,0)

        if self.future:  
            # Future center
            future_center = self.centers[index+1]
            #print('future center', future_center)
            # Y: onehot target (64,64) of future center
            future_onehot = torch.zeros((1, self.canvas_size, self.canvas_size))
            future_onehot[0, future_center[0], future_center[1]] = 1
            y = future_onehot.reshape((-1, self.canvas_size * self.canvas_size))

        else:
            y = onehot.reshape((-1, self.canvas_size * self.canvas_size))

        return center, onehot, y, [speed, angle*360]

