import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.autograd import Variable

from .utility import models, sdr


class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3, 
                 kernel=3, num_spk=2, causal=False):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # TCN separator
        self.TCN = models.TCN(self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                              self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
            batch_size = input.size(0)
            nsample = input.size(2)
            rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input):
        
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        
        # waveform encoder
        enc_output = self.encoder(output)  

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  
        masked_output = enc_output.unsqueeze(1) * masks  
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  
        output = output.view(batch_size, self.num_spk, -1)  
        
        return output

def test_conv_tasnet():
    a = torch.rand(2, 32000)
    nnet = TasNet()
    a = nnet(a)
    s1 = a[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
