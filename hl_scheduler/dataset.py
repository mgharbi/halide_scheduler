import torch as th
from torch.utils.data import Dataset
import numpy as np

class PipelineDataset(Dataset):
  def __init__(self, length=5, ndimensions=3):
    super(PipelineDataset, self).__init__()
    self.length = length
    self.ndimensions = ndimensions

  def __len__(self):
    return 32*5000

  def __getitem__(self, idx):
    tile_sizes = np.random.randint(1, 5, size=(self.length, self.ndimensions)).astype(np.float32)
    inline_or_root = np.random.randint(0, 2, size=(self.length, 1)).astype(np.float32)

    masked = tile_sizes*inline_or_root
    prod = np.product(masked, axis=1)
    cost = np.sum(prod)
    cost = np.expand_dims(cost, 1)

    features = np.concatenate([tile_sizes, inline_or_root], 1)

    features = features.astype(np.float32)
    cost = cost.astype(np.float32)

    # [time/depth, nfeatures]
    return features, cost
