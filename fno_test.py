import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
		
class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x) # [batch_size, 1024, 2]
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
		
################################################################
#  configurations
################################################################
# ntrain = 1000
# ntest = 100
ntrain = 400
ntest = 100

sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

batch_size = 100
learning_rate = 0.001
epochs = 2000
iterations = epochs*(ntrain//batch_size)

modes = 512
#width = 64
width = 64
################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
#dataloader = MatReader('data/burgers_data_R10.mat')
dataloader = MatReader('data/force_disp_map_rand_hs20_2.mat')
# dataloader = MatReader('data/force_disp_map_rand_500_m.mat')
# dataloader = MatReader('data/force_disp_map_rand_gauss_norm_1D.mat')

x_data = dataloader.read_field('a_train')[:,::sub]
y_data = dataloader.read_field('u_train')[:,::sub]

### shuffling the indices of test and train data 
indices = np.arange(len(x_data))
np.random.shuffle(indices)

train_indices = indices[:ntrain]
test_indices = indices[ntrain:ntrain + ntest]

x_train = x_data[train_indices,:]
y_train = y_data[train_indices,:]
x_test = x_data[test_indices,:]
y_test = y_data[test_indices,:]
# x_train = x_data[:ntrain,:]
# y_train = y_data[:ntrain,:]
# x_test = x_data[-ntest:,:]
# y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model
model = FNO1d(modes, width).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

train_ms =[] 
train_L2 =[]
test_L2 =[]

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    print('epoch = \n',ep)
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward() # use the l2 relative loss

        optimizer.step()
        scheduler.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    train_ms.append(train_mse)
    train_L2.append(train_l2)
    test_L2.append(test_l2)

    t2 = default_timer()
	
# for plottting

import matplotlib.pyplot as plt
plt.plot(range(1, len(train_ms) + 1), train_ms, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training MSE')
plt.title('Training MSE over Epochs')
plt.yscale('log')
plt.grid(True)
## file name ##
# filename = f"ntrains_{ntrain}_ntests_{ntest}_modes_{modes}_width_{width}_rand_hs_10.png"
# plt.savefig(filename)
plt.show()
#
plt.plot(range(1, len(train_L2) + 1), train_L2, marker='o',color='black', markersize=8)
# plt.xlabel('Epoch')
plt.ylabel('Training L2', fontsize=24)
# plt.title('Training L2 over Epochs')
# plt.yscale('log')
# plt.grid(True)
# plt.show()
#
plt.plot(range(1, len(test_L2) + 1), test_L2, marker='o', color='red', markersize=8)
plt.xlabel('Epoch', fontsize=24 )
# plt.ylabel('Test L2')
# plt.title('Test L2 over Epochs')
plt.yscale('log')
plt.legend(['Training', 'Test'], fontsize=18)
plt.grid(True)
selected_xticks = [0, 500, 1000, 1500, 2000]  # Specify the positions of the ticks you want to keep
plt.xticks(selected_xticks, fontsize=20)
# plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
## file name ##
filename = f"ntrains_{ntrain}_ntests_{ntest}_modes_{modes}_width_{width}_hs_20.png"
plt.savefig(filename)

plt.show()