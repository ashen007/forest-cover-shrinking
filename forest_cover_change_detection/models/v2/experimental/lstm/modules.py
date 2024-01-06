import torch
from torch.autograd import Variable
from torch import nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.in_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.forget_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.cell_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)

    def forward(self, input, h_state, c_state):
        conc_inputs = torch.cat((input, h_state), 1)

        in_gate = self.in_gate(conc_inputs)
        forget_gate = self.forget_gate(conc_inputs)
        out_gate = self.out_gate(conc_inputs)
        cell_gate = self.cell_gate(conc_inputs)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        c_state = (forget_gate * c_state) + (in_gate * cell_gate)
        h_state = out_gate * torch.tanh(c_state)

        return h_state, c_state


class SetValues(nn.Module):
    def __init__(self, hidden_size, height, width):
        super(SetValues, self).__init__()
        self.hidden_size = int(hidden_size)
        self.height = int(height)
        self.width = int(width)
        self.dropout = nn.Dropout(0.7)
        self.RCell = RNNCell(self.hidden_size, self.hidden_size)

    def forward(self, xinp):
        h_state, c_state = (Variable(torch.zeros(xinp.shape[0], self.hidden_size, self.height, self.width)).cuda(),
                            Variable(torch.zeros(xinp.shape[0], self.hidden_size, self.height, self.width)).cuda())

        h_state, c_state = self.RCell(xinp, h_state, c_state)

        return h_state


if __name__ == '__main__':
    t = torch.randn(4, 16, 64, 64).cuda()
    lstm = SetValues(16, 64, 64).cuda()
    s = lstm(t)

    print(s.shape)
