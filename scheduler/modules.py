import torch as th
import torch.nn as nn
import torch.nn.functional as F

# class CostRNN(nn.Module):
#   def __init__(self, input_size, state_size, output_size, hidden_size=64):
#     super(CostRNN, self).__init__()
#     self.state_size = state_size
#     self.h1 = nn.Linear(input_size+state_size, hidden_size)
#     self.h2 = nn.Linear(hidden_size, hidden_size)
#     self.h3 = nn.Linear(hidden_size, output_size + state_size)
#
#   def forward(self, input, state):
#     x = th.cat([input, state], 1)
#     h1 = F.relu(self.h1(x))
#     h2 = F.relu(self.h2(h1))
#     h3 = self.h3(h2)
#     state = F.tanh(h3[:, :self.state_size, ...])
#     output = h3[:, self.state_size:, ...]
#     return output, state
#
#   def init_state(self, batch_size):
#     return th.zeros((batch_size, self.state_size))

class CostPredictor(nn.Module):
  def __init__(self, input_size, output_size):
    super(CostPredictor, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.h1 = nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.h1(x)

class RelativeError(nn.Module):
  def __init__(self):
    super(RelativeError, self).__init__()
    self.eps = 1e-2

  def forward(self, src, target):
    diff = th.abs(src-target) / (target + self.eps)
    return th.mean(diff)
