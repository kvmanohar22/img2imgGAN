import tensorflow as tf 
import numpy as np
import time
import os
import re
from datetime import datetime

from modules import *
from utils import Dataset

class Model(object):

   def __init__(self, opts, is_training):
      self.opts = opts
      self.sess = tf.Session()
      self.is_training = is_training
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
      self.writer = tf.summary.FileWriter(self.opts.summary_dir, self.sess.graph)

   def placeholders(self):
      """
      Allocate placeholders of the graph
      """
      pass

   def build_graph(self):
      """
      Generate the graph
      """
      pass

   def summaries(self):
      """
      Adds all the necessary summaries
      """
      pass

   def loss(self):
      """
      Implements the loss function
      """
      pass

   def train(self):
      """
      Train the network
      """
      pass

   def checkpoint(self, iteration):
      """
      Creates a checkpoint at the given iteration
      """
      self.saver.save(self.sess, os.path.join(self.opts.summary_dir, "model_{}.ckpt").format(iteration))

   def test_graph(self):
      """
      Generate the graph and check if the connections are correct
      """
      pass
