#!/usr/bin/env python
# -*- coding: utf-8 *-

import os
import argparse
import logging

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

import hl_scheduler.modules as modules
import hl_scheduler.dataset as dataset

import torchlib.viz as viz

log = logging.getLogger("hl_scheduler")

def save(checkpoint, model, params, optimizer, step):
  if os.path.exists(checkpoint):
    backup = os.path.splitext(checkpoint)[0] + "_backup.ph"
    log.info("moving old checkpoint {} to backup {}".format(checkpoint, backup))
    os.rename(checkpoint, backup)
  log.info("saving checkpoint {} at step {}".format(checkpoint, step))
  th.save({
    'model_state': model.state_dict(),
    'params': params,
    'optimizer': optimizer.state_dict(),
    'step': step,
    } , checkpoint)

def main(args):
  bs = args.batch_size
  ndimensions = args.ndims
  length = args.pipeline_length

  if not os.path.exists(args.output):
    log.info("Creating output folder {}".format(args.output))
    os.makedirs(args.output)
  checkpoint = os.path.join(args.output, "checkpoint.ph")

  # RNN metaparams
  nlayers = 3
  state_size = 128

  data = dataset.PipelineDataset(length, ndimensions)
  dataloader = DataLoader(data, batch_size=bs)

  model = modules.PipelineRNN(ndimensions+1, state_size, nlayers)

  if args.cuda:
    model = model.cuda()

  criterion = nn.L1Loss()
  opt = th.optim.Adam(model.parameters(), lr=1e-4)

  rel = modules.RelativeError()

  loss_viz = viz.ScalarVisualizer("loss", env="scheduler")
  rel_viz = viz.ScalarVisualizer("relative_error", env="scheduler")

  global_step = 0
  if os.path.isfile(checkpoint):
    log.info("Resuming from checkpoint {}".format(checkpoint))
    chkpt = th.load(checkpoint)
    model.load_state_dict(chkpt['model_state'])
    opt.load_state_dict(chkpt['optimizer'])
    global_step = chkpt['step']


  smoothed_loss = 0
  smoothed_rel_err = 0
  ema = 0.99
  for epoch in range(1000):
    for step, batch in enumerate(dataloader):
      global_step += 1

      input_data = Variable(batch[0])
      input_data = input_data.permute(1, 0, 2)
      target = Variable(batch[1])

      state = model.initial_state(args.batch_size)

      if args.cuda:
        input_data = input_data.cuda()
        target = target.cuda()
        state = state.cuda()

      model.zero_grad()
      _, cost = model(input_data, state)

      loss = criterion(cost, target)
      loss.backward()
      opt.step()

      rel_err = rel(cost, target)

      if epoch == 0 and step == 0:
        smoothed_loss = loss.data[0]
        smoothed_rel_err = rel_err.data[0]
      else:
        smoothed_loss = ema*smoothed_loss + (1.0-ema)*loss.data[0]
        smoothed_rel_err = ema*smoothed_rel_err + (1.0-ema)*rel_err.data[0]
        # smoothed_rel_err = rel_err.data[0]

      if step % 10 == 0:
        frac = step*1.0/len(dataloader) + epoch
        loss_viz.update(frac, np.log10(smoothed_loss))
        rel_viz.update(frac, np.log10(smoothed_rel_err))

      if step % 100 == 0:
        log.info("Step {}.{} loss = {:.3f}\t rel = {:.3f} predicted={:.3f} target={:.3f}".format(
          epoch, step, smoothed_loss, smoothed_rel_err, cost.data.mean(), target.data.mean()))

    save(checkpoint, model, None, opt, global_step)
    log.info("End of epoch, saving model")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("output", type=str)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--ndims", type=int, default=3)
  parser.add_argument("--pipeline_length", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--cuda", dest="cuda", action="store_true")
  parser.set_defaults(cuda=False)
  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)

  main(args)
