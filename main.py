from functions import *
from architectures import *
from losses import *
from dataset import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

# Generate dataset
ds = Snake(canvas_size=28, square_size=2, speed_channel=True, future=True, square=True, complexity=True)

# Show one example
c, x, y, s = ds[501]
print('x shape =', x.shape)
print('x shape =', y.shape)

# Split train/test partitions
train_quadrant_index, test_quadrant_index = get_quadrant_indexes(ds)

# Wrap transforming function to dataset object
t_ds = TransformedDataset(ds, xy_transform=xy_transform)

# Get training/test subsets
train_ds = Subset(t_ds, indices=train_quadrant_index)
test_ds = Subset(t_ds, indices=test_quadrant_index)

# Configure dataloaders
batch_size = 12
num_workers = 4

trainloader = DataLoader(train_ds, shuffle=True, 
                          batch_size=batch_size, 
                          num_workers=num_workers, 
                          pin_memory=True)

testloader = DataLoader(test_ds, shuffle=True, 
                         batch_size=batch_size, 
                         num_workers=1, 
                         pin_memory=True)

print('train and test dataloaders are ready...')

# Init STCoordConv
net = CoordConv(canvas_size=28, nbr_channels=3).to(device)

# Set optimizers
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
criterion = cross_entropy_one_hot #soft_cross_entropy
epochs = 25

# Train model
train_model(epochs, net, criterion, optimizer, trainloader)

# Eval model
eval_model(net, criterion, testloader)
