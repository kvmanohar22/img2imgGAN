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
      """Class to handle data management

      Args:
         opts: All the hyper-parameters of the network
      """
      self.opts = opts

   def load_batch(self, start_idx, end_idx):
      """Loads a batch of data
      
      Args:
         start_idx: The starting index of the batch
         end_idx  : The ending index of the batch

      Returns:
         ndarray of images and their corresponding labels
      """
      pass
