import io
import os
import sys
import time
import argparse
from time import gmtime, strftime
import tensorflow as tf

from utils import *
from config import FLAGS
import nnet
import utils
import nnet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(_):
   priliminary_checks(FLAGS)
   idx = get_runid(FLAGS)
   create_rundirs(FLAGS, idx)
   dump_model_params(FLAGS)

   net = nnet.Model(FLAGS, FLAGS.train)
   print 'Training the network...'
   net.train()
   print 'Done training the network...'


if __name__ == '__main__':
   try:
      tf.app.run()
   except Exception as E:
      print E
