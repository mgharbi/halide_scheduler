#!/usr/bin/env python
# -*- coding: utf-8 *-

import os
import argparse

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

import scheduler.modules as modules
import scheduler.dataset as dataset

import torchlib.viz as viz

def main(args):
  bs = args.batch_size
  ndimensions = args.ndims
  length = args.pipeline_length

  # RNN metaparams
  nlayers = 3
  state_size = 128

  data = dataset.PipelineDataset(length, ndimensions)
  dataloader = DataLoader(data, batch_size=bs)

  rnn = th.nn.GRU(ndimensions+1, state_size, nlayers)
  cost_predictor = modules.CostPredictor(state_size, 1)

  params = [p for p in rnn.parameters()]
  params += [p for p in cost_predictor.parameters()]

  criterion = nn.L1Loss()
  opt = th.optim.Adam(params, lr=1e-4)

  rel = modules.RelativeError()

  loss_viz = viz.ScalarVisualizer("loss", env="scheduler")
  rel_viz = viz.ScalarVisualizer("relative_error", env="scheduler")

  smoothed_loss = 0
  smoothed_rel_err = 0
  ema = 0.99
  for epoch in range(10):
    for step, batch in enumerate(dataloader):
      input_data = Variable(batch[0])
      input_data = input_data.permute(1, 0, 2)
      target = Variable(batch[1])

      rnn.zero_grad()

      state = Variable(th.zeros((nlayers, bs, state_size)))
      output, state = rnn(input_data, state)
      cost = cost_predictor(output[-1, ...])

      loss = criterion(cost, target)
      loss.backward()
      opt.step()

      rel_err = rel(cost, target)

      # smoothed_loss = loss.data[0]
      if epoch == 0 and step == 0:
        smoothed_loss = loss.data[0]
        smoothed_rel_err = rel_err.data[0]
      else:
        smoothed_loss = ema*smoothed_loss + (1.0-ema)*loss.data[0]
        smoothed_rel_err = ema*smoothed_rel_err + (1.0-ema)*rel_err.data[0]

      frac = step*1.0/len(dataloader) + epoch
      loss_viz.update(frac, smoothed_loss)
      rel_viz.update(frac, np.log10(smoothed_rel_err))

      if step % 100 == 0:
        print("Step {}.{} loss = {:.3f}\t predicted={:.3f}\t target={:.3f}".format(
          epoch, step, smoothed_loss, cost.data[0, 0], target.data[0]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--ndims", type=int, default=3)
  parser.add_argument("--pipeline_length", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--cuda", dest="cuda", action="store_true")
  parser.set_defaults(cuda=False)
  args = parser.parse_args()
  main(args)
