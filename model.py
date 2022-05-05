import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class SMP(nn.Module):
    def __init__(self, num_channels,kernel_size):
        super(SMP, self).__init__()
        self.num_layers = 4
        self.in_channel = num_channels
        self.kernel_size = kernel_size
        self.padding=(self.kernel_size-1)//2
        self.num_filters = 64

        self.layer_in = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,kernel_size=self.kernel_size, padding=self.padding, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_in.weight.data)
        self.lam_in = nn.Parameter(0.01 * torch.ones(1,self.num_filters,1,1))

        self.lam_i = []
        self.layer_down = []
        self.layer_up = []
        for i in range(self.num_layers):
            down_conv = 'down_conv_{}'.format(i)
            up_conv = 'up_conv_{}'.format(i)
            lam_id = 'lam_{}'.format(i)
            layer_2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,kernel_size=self.kernel_size, padding=self.padding, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_2.weight.data)
            setattr(self, down_conv, layer_2)
            self.layer_down.append(getattr(self, down_conv))
            layer_3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,kernel_size=self.kernel_size, padding=self.padding, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_3.weight.data)
            setattr(self, up_conv, layer_3)
            self.layer_up.append(getattr(self, up_conv))

            lam_ = nn.Parameter(0.01 * torch.ones(1,self.num_filters,1,1))
            setattr(self, lam_id, lam_)
            self.lam_i.append(getattr(self, lam_id))

    def forward(self, mod):
        p1 = self.layer_in(mod)
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam_in))

        for i in range(self.num_layers):
            p3 = self.layer_down[i](tensor)
            p4 = self.layer_up[i](p3)
            p5 = tensor - p4
            p6 = torch.add(p1, p5)
            tensor = torch.mul(torch.sign(p6), F.relu(torch.abs(p6) - self.lam_i[i]))
        return tensor

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.channel = 3
        self.filters = 64
        self.decoconv1 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=7,stride=1, padding=3, bias=False)
        nn.init.xavier_uniform_(self.decoconv1.weight.data)
        self.decoconv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=5,stride=1, padding=2, bias=False)
        nn.init.xavier_uniform_(self.decoconv2.weight.data)
        self.decoconv3 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=3,stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.decoconv3.weight.data)
        
    def forward(self, z1,z2,z3):
    
        rec_x1 = self.decoconv1(z1)
        rec_x2 = self.decoconv2(z2)
        rec_x3 = self.decoconv3(z3)
        rec_x = rec_x1 + rec_x2 + rec_x3
        return rec_x,rec_x1,rec_x2,rec_x3

class MCSCNet(nn.Module):
    def __init__(self):
        super(MCSCNet, self).__init__()
        self.channel = 3
        self.num_filters = 64
        self.kernel_size = 7

        self.SMP1 = SMP(num_channels=self.channel,kernel_size=7)
        self.conv1 = nn.Conv2d(in_channels=self.num_filters,out_channels=self.channel,kernel_size=7,stride=1,padding=3,bias=False)
        self.SMP2 = SMP(num_channels=self.channel, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=5, stride=1,padding=2, bias=False)
        self.SMP3 = SMP(num_channels=self.channel, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=3, stride=1,padding=1, bias=False)

        nn.init.xavier_uniform_(self.conv1.weight.data)
        nn.init.xavier_uniform_(self.conv2.weight.data)
        nn.init.xavier_uniform_(self.conv3.weight.data)

        self.SMP4 = SMP(num_channels=self.channel, kernel_size=7)
        self.conv4 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=7, stride=1,padding=3, bias=False)
        self.SMP5 = SMP(num_channels=self.channel, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=5, stride=1,padding=2, bias=False)
        self.SMP6 = SMP(num_channels=self.channel, kernel_size=3)

        nn.init.xavier_uniform_(self.conv4.weight.data)
        nn.init.xavier_uniform_(self.conv5.weight.data)

        self.decoder=decoder()


    def forward(self, x):
        # The first round
        z1_first=self.SMP1(x)
        x1_first=self.conv1(z1_first)
        x2hat_first=x-x1_first
        z2_first=self.SMP2(x2hat_first)
        x2_first=self.conv2(z2_first)
        x3hat_first=x2hat_first-x2_first
        z3_first=self.SMP3(x3hat_first)
        x3_first=self.conv3(z3_first)
        x1_hat=x-x3_first-x2_first
        # The second round
        z1=self.SMP4(x1_hat)
        x1=self.conv4(z1)
        x2hat=x-x1
        z2=self.SMP5(x2hat)
        x2=self.conv5(z2)
        x3hat=x2hat-x2
        z3=self.SMP6(x3hat)
        f_pred,x1_pred,x2_pred,x3_pred=self.decoder(z1,z2,z3)
        return f_pred,z1,z2,z3
if __name__ == '__main__':
    model = MCSCNet()
    macs, params = get_model_complexity_info(model, (3, 112,112), as_strings=True,print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print(MS)
    # a = torch.rand([10, 3, 64, 64])
    # a_out=MS(a)
    # print(a_out.shape)