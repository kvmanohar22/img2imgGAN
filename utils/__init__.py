from six.moves import cPickle
import numpy as np
import os
import time
from skimage import io
from skimage.transform import resize
from scipy.misc import imsave

import utils
from utils import *

class Dataset(object):

   def __init__(self, opts):
      self.opts = opts

   def load_batch(self, start_idx, end_idx):
      """
      Loads a batch of data
      """
      pass
