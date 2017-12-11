import numpy as np
import os
from skimage import io
from skimage.transform import resize
from scipy.misc import imsave


def priliminary_checks(flags):
   """
   Checks the existance of directories and creates them if necessary
   """
   log_dir = os.path.join(flags.root_dir, 'logs')
   if not os.path.exists(log_dir):
      os.makedirs(log_dir)
   # if not os.path.exists(flags.dataset_dir):
   #    print 'Dataset direcstory not found !\nExiting...'
   #    exit()
   if not os.path.exists(flags.sample_dir):
      os.makedirs(flags.sample_dir)
   if not os.path.exists(flags.summary_dir):
      os.makedirs(flags.summary_dir)

def create_rundirs(flags, id):
   """
   Creates the directories for the `id`th run of the model
   """
   os.makedirs(os.path.join(flags.sample_dir, 'Run_{}'.format(id)))
   os.makedirs(os.path.join(flags.summary_dir, 'Run_{}'.format(id)))
   flags.summary_dir = os.path.join(flags.summary_dir, 'Run_{}'.format(id))
   flags.sample_dir = os.path.join(flags.sample_dir, 'Run_{}'.format(id))

def dump_model_params(flags):
   """
   Writes model params to a file
   """
   idx = get_runid(flags)
   # TODO : Add the details of the model

def get_runid(flags):
   """
   Returns the number of the present run of the model
   """
   summary_dir = flags.summary_dir
   dirs = os.listdir(summary_dir)
   return len(dirs)+1

def imread(img_path):
   """
   Reads an image
   """
   img = io.imread(img_path)
   return img
