from six.moves import cPickle
import numpy as np
import os
import sys
import time
from skimage import io
from skimage.transform import resize
from scipy.misc import imsave

import utils
from utils import *

checker = lambda dir, data, dtype: True if os.path.exists(os.path.join(\
                            dir, data, '{}.npy'.format(dtype))) else False
loader  = lambda dir, data, dtype: np.load(os.path.join(dir, data,\
                            '{}.npy'.format(dtype)))

class Dataset(object):

   def __init__(self, opts, load=False):
      """Class to handle data management

      Args:
         opts: All the hyper-parameters of the network
         load: Load dataset if this is true
      """
      self.records = {}
      self.opts = opts
      self.check_records()
      if load:
         if opts.load_images:
            raise NotImplementedError("Cannot load images before runtime")
         else:
            # Handle the case for validating and testing
            file_name = os.path.join(opts.dataset_dir, opts.dataset, 'train.txt')
            with open(file_name) as f:
               lines = f.readlines()
               lines = [line.strip() for line in lines]
               self.t_image_paths = np.array(lines)

   def train_size(self):
      return len(self.t_image_paths)

   def load_batch(self, start_idx, end_idx):
      """Loads a batch of data
      
      Args:
         start_idx: First index in the list
         end_idx  : End index in the list

      Returns:
         ndarray of images
      """
      if self.opts.load_images:
         raise NotImplementedError("Cannot load images")
      else:
         return self.load_images_from_files(start_idx, end_idx)

   def load_data(self, dataset=None):
      """Loads a specific dataset

      Args:
         dataset: Name of the dataset
      """
      if dataset is None:
         raise ValueError("Must specify dataset name")

      try:
         if self.records[dataset]:
            print 'Loading {} data records...'.format(dataset)
            self.train_data_in = loader(self.opts.dataset_dir, dataset, 'train_in')
            self.train_data_out = loader(self.opts.dataset_dir, dataset, 'train_out')
            self.val_data_in = loader(self.opts.dataset_dir, dataset, 'val_in')
            self.val_data_out = loader(self.opts.dataset_dir, dataset, 'val_out')
      except KeyError as E:
         raise KeyError('Dataset \"{}\" doesnot exist'.format(dataset))

   def load_images_from_files(self, start_idx, end_idx):
      """Loads images by reading images from files
      
      Args:
         start_idx: First index in the list
         end_idx  : End index in the list

      Returns:
         Images as pairs
      """
      images_A = np.empty([self.opts.batch_size, img_dim, img_dim, 3], dtype=np.uint8)
      images_B = np.empty([self.opts.batch_size, img_dim, img_dim, 3], dtype=np.uint8)
      for idx, paths in enumerate(self.t_image_paths[start_idx:end_idx]):
         path = os.path.join(self.opts.dataset_dir, self.opts.dataset,
                             paths.split(' ').strip())
         try:
            image = io.imread(path)
         except IOError:
            raise IOError("Cannot read the image {}" % path)

         split_len = 600 if self.opts.dataset == 'maps' else 256
         images_A[idx] = image[:, :split_len, :]
         images_B[idx] = image[:, split_len:, :]
      return images_A, images_B

   def create_records(self, dataset):
      """Creates numpy records

      Args:
         dataset: Name of the dataset
      """
      base_path = os.path.join(self.opts.dataset_dir, dataset)
      train_files = os.listdir(os.path.join(base_path, 'train'))
      test_files = os.listdir(os.path.join(base_path, 'val'))

      print 'Creating numpy records for {} train data'.format(dataset)
      t_data_in, t_data_out, flag = self.create_numpy_records(dataset, train_files, "train")
      if flag:
         sys.stdout.write('\nSaving the file...\n')
         sys.stdout.flush()
         self.save_records(dataset, t_data_in, "train_in")
         self.save_records(dataset, t_data_out, "train_out")
      print '\nCreating numpy records for {} validation data'.format(dataset)
      v_data_in, v_data_out, flag = self.create_numpy_records(dataset, test_files, "val")
      if flag:
         sys.stdout.write('\nSaving the file...\n')
         sys.stdout.flush()
         self.save_records(dataset, v_data_in, "val_in")
         self.save_records(dataset, v_data_out, "val_out")

   def create_numpy_records(self, dataset, files, dtype):
      """Creates numpy files of the given image paths

      Args:
         dataset: Name of the dataset
         files  : List of image paths
         dtype  : "train" or "val"

      Returns:
         Numpy records of the given dataset
      """
      if os.path.exists(os.path.join(self.opts.dataset_dir, dataset, '{}.npy'.format(dtype))):
         print 'Numpy records already exists for "{}" {} dataset'.format(dataset, dtype)
         return None, False

      done_len_size = 30
      split_len = 600 if dataset == 'maps' else 256
      img_dim = split_len
      paths = [os.path.join(self.opts.dataset_dir, dataset,
         dtype, file) for file in files]
      
      data_in  = np.empty([len(files), img_dim, img_dim, 3], dtype=np.uint8)
      data_out = np.empty([len(files), img_dim, img_dim, 3], dtype=np.uint8)
      
      for idx, file in enumerate(paths):
         sys.stdout.write('\r')
         percentage = (idx+1.)/len(files)
         done_len = int(percentage*done_len_size)
         args = [done_len*'=', (done_len_size-done_len-1)*' ', percentage*100]
         sys.stdout.write('[{}>{}]{:.0f}%'.format(*args))
         sys.stdout.flush()

         img = io.imread(file)
         img1, img2 = img[:, :split_len, :], img[:, split_len:, :]
         data_in[idx], data_out[idx] = img1, img2
      return data_in, data_out, True

   def save_records(self, dataset, data, dtype):
      """Saves the numpy record files for the given dataset

      Args:
         dataset: Name of the dataset
         data   : Numpy array containing data
         dtype  : "train" or "val"
      """
      np.save(os.path.join(self.opts.dataset_dir, dataset, dtype), data)

   def check_records(self):
      """Checks for numpy records
      """
      self.records['edges2shoes'] = checker(self.opts.dataset_dir, 'edges2shoes', 'train')
      self.records['edges2handbags'] = checker(self.opts.dataset_dir, 'edges2handbags', 'train')
      self.records['facades'] = checker(self.opts.dataset_dir, 'facades', 'val')
      self.records['maps'] = checker(self.opts.dataset_dir, 'maps', 'train')
