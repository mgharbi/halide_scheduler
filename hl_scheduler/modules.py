import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PipelineRNN(nn.Module):
  def __init__(self, input_size, state_size, nlayers):
    super(PipelineRNN, self).__init__()
    self.input_size = input_size
    self.state_size = state_size
    self.nlayers = nlayers

    self.rnn = th.nn.GRU(input_size, state_size, nlayers)
    self.cost_prediction = nn.Linear(state_size, 1)


  def initial_state(self, bs):
    state = th.zeros(self.nlayers, bs, self.state_size)
    return Variable(state)

  def forward(self, input_data, state):
    output, state = self.rnn(input_data, state)
    return state, self.cost_prediction(output[-1, ...])  # use state of last time step as features


class RelativeError(nn.Module):
  def __init__(self):
    super(RelativeError, self).__init__()
    self.eps = 1e-2

  def forward(self, src, target):
    diff = th.abs(src-target) / (th.abs(target) + self.eps)
    return th.mean(diff)
