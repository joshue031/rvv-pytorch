"""
rvv_fcnet.py

An FC net for use with RVV commands.
"""
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

netdebug = False

# Load the backend.
torch.ops.load_library("build/librvv_pytorch.so")

# The network.
class rvvFC(nn.Module):
    def __init__(self, n_hidden):
        super(rvvFC, self).__init__()

        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(1, n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,1)

    def forward(self, x):
        if(netdebug): print(x.shape)
        x = F.relu(self.fc1(x))
        if(netdebug): print(x.shape)
        x = F.relu(self.fc2(x))
        if(netdebug): print(x.shape)
        x = F.relu(self.fc3(x))

        return x

# The dataset class.
class sqDataset(Dataset):
    def __init__(self,length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        x = np.random.random()
        xsq = x**2
        return [x,xsq]

# Train function.
def train(model, epoch, train_loader, optimizer):

    losses_epoch = []; accuracies_epoch = []
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)
        data, target = data.unsqueeze(1).float(), target.unsqueeze(1).float()
        optimizer.zero_grad()

        #print("Running batch",batch_idx,"through...")
        output_score = model(data)
        #print("Computing MSE loss....")
        m = nn.MSELoss()
        loss = m(output_score,target)

        #print("Backward step...")
        loss.backward()
        optimizer.step()

        #print("Computing accuracy...")
        correctvals = abs(output_score-target) < 1e-3
        accuracy = correctvals.sum().float() / float(target.size(0))

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}; Accuracy {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), output_score.data.max(), output_score.data.min(), accuracy.data.item()))

        losses_epoch.append(loss.data.item())
        accuracies_epoch.append(accuracy.data.item())

    print("---EPOCH AVG TRAIN LOSS:",np.mean(losses_epoch),"ACCURACY:",np.mean(accuracies_epoch))
    with open("train.txt", "a") as ftrain:
        ftrain.write("{} {} {}\n".format(epoch,np.mean(losses_epoch),np.mean(accuracies_epoch)))

    return np.mean(losses_epoch)

modeldir = 'models'
lrate       = 1e-4   # Learning rate to use in the training.
load_model  = False  # Load an existing model
epoch_start = 0      # Number of initial epoch
epoch_end   = 20    # Number of final epoch
model_load_checkpoint = "{}/model_0.pt".format(modeldir)

# Create a new dataset.
dset = sqDataset(1000)

# Create the loaders.
train_loader = DataLoader(dset, batch_size=100, shuffle=False, num_workers=1)

# Define the model.
model = rvvFC(n_hidden=200)
#model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=lrate, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Load the model from file.
if(load_model):
    model.load_state_dict(torch.load(model_load_checkpoint))
    #model.load_state_dict(torch.load(model_load_checkpoint,map_location=torch.device('cpu')))
    model.eval()


# Run the training.
for epoch in range(epoch_start,epoch_end):
    print("Epoch: ", epoch)
    model.train()
    train_loss = train(model, epoch, train_loader, optimizer)
    scheduler.step(train_loss)
    #if(epoch % 50 == 0):
    torch.save(model.state_dict(), "{}/model_{}.pt".format(modeldir,epoch))

# Run some evaluations.
model.eval()
x = torch.Tensor(np.arange(0,1.1,0.1)).unsqueeze(1)
xsq = model(x)
print("Input:",x)
print("Output:",xsq)
print("Diff:",xsq-x**2)

print("Done.")
