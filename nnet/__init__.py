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
      """Initialize the model

      Args:
         opts: All the hyper-parameters of the network
         is_training: Boolean which indicates whether the model is in train
            mode or test mode
      """
      self.opts = opts
      self.h = opts.h
      self.w = opts.w
      self.c = opts.c
      self.sess = tf.Session()
      self.should_train = is_training
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
      self.writer = tf.summary.FileWriter(self.opts.summary_dir, self.sess.graph)
      self.build_graph()

   def build_graph(self):
      """Generate various parts of the graph
      """
      self.placeholders()
      self.G  = self.generator(self.input_images, self.code)
      self.D  = self.discriminator(self.input_images, reuse=False)
      self.D_ = self.discriminator(self.G, reuse=True)
      self.summaries()

   def placeholders(self):
      """Allocate placeholders of the graph
      """
      self.input_images = tf.placeholder(tf.float32, [None, self.h, self.w, self.c], name="input_images")
      self.target_images = tf.placeholder(tf.float32, [None, self.h, self.w, self.c], name="target_images")
      self.code = tf.placeholder(tf.float32, [None, self.opts.code_len], name="code")
      self.is_training = tf.placeholder(tf.bool, name="is_training")

   def summaries(self):
      """Adds all the necessary summaries
      """
      in_images = tf.summary.image('Input_images', self.input_images, max_outputs=10)
      tr_images = tf.summary.image('Target_images', self.target_images, max_outputs=10)
      gen_images = tf.summary.image('Gen_images', self.G, max_outputs=10)
      self.summaries = tf.summary.merge_all()

   def encoder(self, image):
      """Encoder which generates the latent code
      
      Args:
         image: Image which is to be encoded

      Returns:
         The encoded latent code
      """

      def normal_encoder(image):
         """Few convolutional layers followed by downsamplinga layers
         """
         pass

      def resnet_encoder(image):
         """Residual Network with several residual blocks
         """
         pass

      if self.opts.encoder_type == "normal":
         return normal_encoder(image)
      else:
         return resnet_encoder(image)

   def generator(self, image, z):
      """Generator graph of GAN

      Args:
         image: Conditioned image on which the generator generates the image 
         z:      latent space code

      Returns:
         Generated image
      """

   def discriminator(self, image, reuse=False):
      """Discriminator graph of GAN
      Authors claim not conditioning the discriminator yields better results
         and hence not conditioning the discriminator with the input image
      Authors also claim that using two discriminators for cVAE-GAN and cLR-GAN
         yields better results, here we share the weights for both of them

      Args:
         image: Input image to the discriminator
         reuse: Flag to check whether to reuse the variables created for the
            discriminator graph

      Returns:
         Whether or not the input image is real or fake
      """

   def loss(self):
      """Implements the loss graph
      """
      def l1_loss(z1, z2):
         """Implements L1 loss graph
         
         Args:
            z1: Image in case of cVAE-GAN
                Vector in case of cLR-GAN
            z2: Image in case of cVAE-GAN
                Vector in case of cLR-GAN

         Returns:
            L1 loss
         """
         pass

      def kl_divergence(p1, p2):
         """Apply KL divergence
         
         Args:
            p1: 1st probability distribution
            p2: 2nd probability distribution

         Returns:
            KL Divergence between the given distributions
         """
         pass

   def train(self):
      """Train the network
      """
      pass

   def checkpoint(self, iteration):
      """Creates a checkpoint at the given iteration

      Args:
         iteration: Iteration number of the training process
      """
      self.saver.save(self.sess, os.path.join(self.opts.summary_dir, "model_{}.ckpt").format(iteration))

   def test_graph(self):
      """Generate the graph and check if the connections are correct
      """
      pass
