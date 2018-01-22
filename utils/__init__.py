from six.moves import cPickle
import numpy as np
import os
import sys
import time
from skimage import io
from skimage.transform import resize
from scipy.misc import imsave

import utils

checker = lambda dir, data, dtype: True if os.path.exists(os.path.join(
                            dir, data, '{}.npy'.format(dtype))) else False
loader  = lambda dir, data, dtype: np.load(os.path.join(dir, data,
                            '{}.npy'.format(dtype)))

class Dataset(object):
   """Helper class for the dataset pipeline
   """

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
            print ' - Loading numpy raw images...'
            begin_time = time.time()
            self.t_images_data = np.load(os.path.join(opts.dataset_dir,
                                                      opts.dataset,
                                                      'train.npy'))
            print ' - Loaded {} images in: {} seconds'.format(len(self.t_images_data)*2,\
               time.time()-begin_time)
         else:
            self.t_image_paths = utils.read_file_lines(
                    os.path.join(opts.dataset_dir,
                                 opts.dataset,
                                 'train.txt'))
            self.v_image_paths = utils.read_file_lines(
                    os.path.join(opts.dataset_dir,
                                 opts.dataset,
                                 'val.txt'))
            test_path = os.path.join(opts.dataset_dir, opts.dataset, 'test.txt')
            if utils.path_exists(test_path):
               self.test_image_paths = utils.read_file_lines(
                       os.path.join(opts.dataset_dir,
                                    opts.dataset,
                                    'test.txt'))

   def train_size(self):
      """Returns the train datasize

      Returns:
         Size of the training dataset
      """
      try:
         length = len(self.t_image_paths)
      except:
         length = len(self.t_images_data)
      return length

   def load_val_batch(self):
      """Loads validation images"""

      img_dim = self.opts.h
      images_A = np.empty([self.opts.batch_size, img_dim, img_dim, 3], dtype=np.float32)
      images_B = np.empty([self.opts.batch_size, img_dim, img_dim, 3], dtype=np.float32)
      for idx, path in enumerate(self.v_image_paths[:self.opts.batch_size]):
         path = os.path.join(self.opts.dataset_dir, self.opts.dataset, 'val', path)
         try:
            image = utils.imread(path)
            image = utils.normalize_images(images=image)
         except IOError:
            raise IOError("Cannot read the image {}" % path)

         split_len = 600 if self.opts.dataset == 'maps' else 256
         images_A[idx] = image[:, :split_len, :]
         images_B[idx] = image[:, split_len:, :]

      return images_A, images_B
      

   def load_batch(self, start_idx, end_idx):
      """Loads a batch of data
      
      Args:
         start_idx: First index in the list
         end_idx  : End index in the list

      Returns:
         ndarray of images
      """
      if start_idx == 0:
         try:
            np.random.shuffle(self.t_image_paths)
         except:
            np.random.shuffle(self.t_images_data)
      if self.opts.load_images:
         return self.load_images_from_numpy_records(start_idx, end_idx)
      else:
         return self.load_images_from_files(start_idx, end_idx)

   def load_images_from_numpy_records(self, start_idx, end_idx):
      """Loads images from numpy records

      Args:
         start_idx: First index in the list
         end_idx  : End index in the list

      Returns:
         Images as pairs
      """
      images_A = utils.normalize_images(self.t_images_data[start_idx:end_idx, 0].astype(np.float32))
      images_B = utils.normalize_images(self.t_images_data[start_idx:end_idx, 1].astype(np.float32))
      return images_A, images_B

   def load_images_from_files(self, start_idx, end_idx):
      """Loads images by reading images from files
      
      Args:
         start_idx: First index in the list
         end_idx  : End index in the list

      Returns:
         Images as pairs
      """
      img_dim = self.opts.h
      images_A = np.empty([self.opts.batch_size, img_dim, img_dim, 3], dtype=np.float32)
      images_B = np.empty([self.opts.batch_size, img_dim, img_dim, 3], dtype=np.float32)
      for idx, path in enumerate(self.t_image_paths[start_idx:end_idx]):
         # TODO: Generalize this method to load test/val images
         path = os.path.join(self.opts.dataset_dir, self.opts.dataset, 'train', path)
         try:
            image = utils.imread(path)
            image = utils.normalize_images(images=image)
         except IOError:
            raise IOError("Cannot read the image {}" % path)

         split_len = 600 if self.opts.dataset == 'maps' else 256
         images_A[idx] = image[:, :split_len, :]
         images_B[idx] = image[:, split_len:, :]

      return images_A, images_B

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

   def create_records(self, dataset):
      """Creates numpy records

      Args:
         dataset: Name of the dataset
      """
      base_path = os.path.join(self.opts.dataset_dir, dataset)
      train_files = os.listdir(os.path.join(base_path, 'train'))
      test_files = os.listdir(os.path.join(base_path, 'val'))

      print 'Creating numpy records for {} train data'.format(dataset)
      t_data, flag = self.create_numpy_records(dataset, train_files, "train")
      if flag:
         sys.stdout.write('\nSaving the file...\n')
         sys.stdout.flush()
         self.save_records(dataset, t_data, "train")
      print '\nCreating numpy records for {} validation data'.format(dataset)
      v_data, flag = self.create_numpy_records(dataset, test_files, "val")
      if flag:
         sys.stdout.write('\nSaving the file...\n')
         sys.stdout.flush()
         self.save_records(dataset, v_data, "val")

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
      
      data = np.empty([len(files), 2, img_dim, img_dim, 3], dtype=np.uint8)

      for idx, file in enumerate(paths):
         sys.stdout.write('\r')
         percentage = (idx+1.)/len(files)
         done_len = int(percentage*done_len_size)
         args = [done_len*'=', (done_len_size-done_len-1)*' ', percentage*100]
         sys.stdout.write('[{}>{}]{:.0f}%'.format(*args))
         sys.stdout.flush()

         img = io.imread(file)
         img1, img2 = img[:, :split_len, :], img[:, split_len:, :]
         data[idx, 0], data[idx, 1] = img1, img2
      return data, True

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
