from torch import nn
import copy
import torch
from torch.autograd import Variable

# Nature_DQN or Double_DQN using CNN
# refer to https://github.com/yfeng997/MadMario
class MarioNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input):
        #print('self.net(input):',self.net(input))
        return self.net(input)

# Nature_DQN or Double DQN using RNN
# refer to https://www.dandelioncloud.cn/article/details/1469891088759758850
class MarioNet_RNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.rnn1 = nn.LSTM(
            input_size=84,
            hidden_size=16,
            num_layers=1, #2
            batch_first=True,  # （time_step,batch,input）时是Ture

            #input_size=84,
            #hidden_size=128,
            #num_layers=2,
            #batch_first=True,  # （time_step,batch,input）时是Ture
        )


        self.out = nn.Linear(16, output_dim)
        #self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(-1, 84, 84)
        r_out, (h_n, h_c) = self.rnn1(x)
        out = self.out(r_out[:, -1, :])
        out = out[0, :]
        out = out.cpu().detach().numpy()
        out = [list(out)]
        out = torch.Tensor(out)
        #print('out:',out)

        return out

# Dueling_DQN using CNN
# refer to https://github.com/likemango/DQN-mario-xiaoyao
class MarioNet_Dueling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.commonLayer = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
            )
        
        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            )

    def forward(self, x):

        x = self.commonLayer(x)
        advantage = self.advantage(x)
        value = self.value(x)
        #print('value:',value)
        return advantage + value - advantage.mean()






