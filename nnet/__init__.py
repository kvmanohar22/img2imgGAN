import tensorflow as tf 
import numpy as np
import time
import datetime
import os
import sys

import re
from datetime import datetime

from modules import *
from utils import Dataset

class Model(object):

   def __init__(self, opts, is_training):
      """Initialize the model by creating various parts of the graph

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
      self.train_mode = is_training
      self.init = tf.global_variables_initializer()
      self.build_graph()

   def build_graph(self):
      """Generate various parts of the graph
      """
      self.non_lin = {'relu' : lambda x: relu(x, name='relu'),
                      'lrelu': lambda x: lrelu(x, name='lrelu'),
                      'tanh' : lambda x: tanh(x, name='tanh')
                     }
      self.placeholders()
      self.E  = self.encoder(self.target_images, self.opts.e_layers, self.opts.e_kernels,
         self.opts.e_nonlin, norm=self.opts.e_norm, reuse=False)
      self.G  = self.generator(self.input_images, self.code, self.opts.g_layers,
         self.opts.g_kernels, self.opts.g_nonlin, norm=self.opts.g_norm)
      self.D  = self.discriminator(self.input_images, self.opts.d_kernels, self.opts.d_layers,
         non_lin=self.opts.d_nonlin, norm=self.opts.d_norm, use_sigmoid=self.opts.d_sigmoid,
         reuse=False)
      self.D_ = self.discriminator(self.G, self.opts.d_kernels, self.opts.d_layers, 
         non_lin=self.opts.d_nonlin, norm=self.opts.d_norm, use_sigmoid=self.opts.d_sigmoid,
         reuse=True)
      self.summaries()
      self.d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
      self.ge_vars = [var for var in tf.trainable_variables() if 'generator' or 'encoder' in var.name]
      # self.model_loss()
      # self.GE_opt = tf.train.AdamOptimizer(self.opts.base_lr).minimize(self.loss['gen_enc'], var_list=self.ge_vars)
      # self.D_opt = tf.train.AdamOptimizer(self.opts.base_lr).minimize(self.loss['dis'], var_list=self.d_vars)

   def placeholders(self):
      """Allocate placeholders of the graph
      """
      self.images_A = tf.placeholder(tf.float32, [None, self.h, self.w, self.c], name="images_A")
      self.images_B = tf.placeholder(tf.float32, [None, self.h, self.w, self.c], name="images_B")
      self.code = tf.placeholder(tf.float32, [None, self.opts.code_len], name="code")
      self.is_training = tf.placeholder(tf.bool, name="is_training")
      if self.opts.direction == 'a2b':
         self.input_images  = self.images_A
         self.target_images = self.images_B
      elif self.opts.direction == 'b2a':
         self.input_images  = self.images_B
         self.target_images = self.images_A
      else:
         raise ValueError("There is no such image transition type")

   def summaries(self):
      """Adds all the necessary summaries
      """
      images_A = tf.summary.image('images_A', self.images_A, max_outputs=10)
      images_B = tf.summary.image('images_B', self.images_B, max_outputs=10)
      gen_images = tf.summary.image('Gen_images', self.G, max_outputs=10)
      self.summaries = tf.summary.merge_all()

   def encoder(self, image, num_layers=3, kernels=64, non_lin='lrelu', norm=None,
               reuse=False):
      """Encoder which generates the latent code
      
      Args:
         image     : Image which is to be encoded
         num_layers: Non linearity to the intermediate layers of the network
         kernels   : Number of filters for the first layer of the network
         non_lin   : Type of non-linearity activation
         norm      : Should use batch normalization
         reuse     : Should reuse the variables?

      Returns:
         The encoded latent code
      """
      self.e_layers = {}
      with tf.variable_scope('encoder'):
         if self.opts.e_type == "normal":
            return self.normal_encoder(image, num_layers=num_layers, output_neurons=8,
               kernels=kernels, non_lin=non_lin, norm=norm, reuse=reuse)
         elif self.opts.e_type == "residual":
            return self.resnet_encoder(image, num_layers, output_neurons=8,
               kernels=kernels, non_lin=non_lin)
         else:
            raise ValueError("No such type of encoder exists!")

   def normal_encoder(self, image, num_layers=4, output_neurons=1, kernels=64, non_lin='lrelu',
                      norm=None, reuse=False):
      """Few convolutional layers followed by downsampling layers
      """
      k, s = 4, 2
      try:
         self.e_layers['conv0'] = conv2d(image, ksize=k, out_channels=kernels*1, stride=s, name='conv0',
            non_lin=self.non_lin[non_lin])
      except KeyError:
         raise KeyError("No such non-linearity is available!")
      for idx in range(1, num_layers):
         input_layer = self.e_layers['conv{}'.format(idx-1)]
         factor = min(2**idx, 4)
         if not norm:
            self.e_layers['conv{}'.format(idx)] = conv2d(input_layer, ksize=k,
               out_channels=kernels*factor, stride=s, name='conv{}'.format(idx),
               non_lin=self.non_lin[non_lin], reuse=reuse)
         else:
            self.e_layers['conv{}'.format(idx)] = conv_bn_lrelu(input_layer, ksize=k,
               out_channels=kernels*factor, is_training=self.train_mode, stride=s,
               name='conv{}'.format(idx), reuse=reuse)

      self.e_layers['pool'] = average_pool(self.e_layers['conv{}'.format(num_layers-1)],
         ksize=8, strides=8, name='pool')
      units = int(np.prod(self.e_layers['pool'].get_shape().as_list()[1:]))
      reshape_layer = tf.reshape(self.e_layers['pool'], [-1, units])
      self.e_layers['full'] = fully_connected(reshape_layer, output_neurons, name='full')
      return self.e_layers['full']

   def resnet_encoder(self, image, num_layers=4, output_neurons=1, kernels=64, non_linearity='relu'):
      """Residual Network with several residual blocks
      """
      raise NotImplementedError("Not Implemented")

   def generator(self, image, z, layers=3, kernels=64, non_lin='relu', norm=None,
                 reuse=False):
      """Generator graph of GAN

      Args:
         image  : Conditioned image on which the generator generates the image 
         z      : Latent space code
         layers : The number of layers either in downsampling / upsampling 
         kernels: Number of kernels to the first layer of the network
         non_lin: Non linearity to be used
         norm   : Whether to use batch normalization layer
         reuse  : Whether to reuse the variables created for generator graph

      Returns:
         Generated image
      """
      self.g_layers = {}
      with tf.variable_scope('generator'):
         if self.opts.where_add == "input":
            return self.generator_input(image, z, layers, kernels, non_lin, norm,
                                        reuse)
         elif self.opts.where_add == "all":
            return self.generator_all(image, z, layers, kernels, non_lin, norm,
                                      reuse)
         else:
            raise ValueError("No such type of generator exists!")

   def generator_input(self, image, z, layers=3, kernels=64, non_lin='lrelu', norm=None,
                       reuse=False):
      """Generator graph where noise is concatenated to the first layer
      """

      with tf.name_scope('replication'):
         tiled_z = tf.tile(self.code, [self.opts.batch_size, self.w*self.h], name='tiling')
         reshaped = tf.reshape(tiled_z, [-1, self.h, self.w, self.opts.code_len], name='reshape')
         in_layer = tf.concat([image, reshaped], axis=3, name='concat')
      k, s = 4, 2
      factor = 1
      # Downsampling
      for idx in xrange(layers):
         factor = min(2**idx, 4)
         if not norm:
            self.g_layers['conv{}'.format(idx)] = conv2d(in_layer, ksize=k, 
               out_channels=kernels*factor, stride=s, name='conv{}'.format(idx),
               non_lin=self.non_lin[non_lin])
         else:
            self.g_layers['conv{}'.format(idx)] = conv_bn_relu(in_layer, ksize=k, 
               out_channels=kernels*factor, is_training=self.train_mode, stride=s,
               name='conv{}'.format(idx), reuse=reuse)
            
         # Maybe add Pooling layer ?
         in_layer = self.g_layers['conv{}'.format(idx)]

      # Upsampling
      in_layer = self.g_layers['conv{}'.format(layers-1)]
      new_idx = layers
      for idx in xrange(layers-2, -1, -1):
         out_shape = self.g_layers['conv{}'.format(idx)].get_shape().as_list()[1]
         out_channels = self.g_layers['conv{}'.format(idx)].get_shape().as_list()[-1]
         self.g_layers['conv{}'.format(new_idx)] = deconv(in_layer, ksize=k,
            out_shape=out_shape, out_channels=out_channels, name='deconv{}'.format(new_idx),
            non_lin=self.non_lin[non_lin], batch_size=self.opts.batch_size)
         input_layer = self.g_layers['conv{}'.format(new_idx)]
         self.g_layers['conv{}'.format(new_idx)] = add_layers(input_layer,
            self.g_layers['conv{}'.format(idx)])
         in_layer = self.g_layers['conv{}'.format(new_idx)]
         new_idx += 1
      self.g_layers['conv{}'.format(layers*2-1)] = deconv(self.g_layers['conv{}'.format(new_idx-1)],
         ksize=k, out_shape=self.h, out_channels=3, name='deconv{}'.format(new_idx),
         non_lin=self.non_lin['tanh'], batch_size=self.opts.batch_size)
      return self.g_layers['conv{}'.format(new_idx)]

   def generator_all(self, image, z, layers=3, kernels=64, non_lin='lrelu', norm=None,
                     reuse=False):
      """Generator graph where noise is to all the layers

      Args:
         image : Conditioned image on which the generator generates the image
         z     : Latent space noise
         layers: The number of layers either in downsampling / upsampling 

      Returns:
         Generated image
      """
      raise NotImplementedError("Not Implemented")


   def discriminator(self, image, kernels=64, num_layers=3, norm_layer=None, non_lin='lrelu', 
                     use_sigmoid=False, reuse=False, norm=None):
      """Discriminator graph of GAN
      The discriminator is a PatchGAN discriminator which consists of two 
         discriminators for two different scales i.e, 70x70 and 140x140
      Authors claim not conditioning the discriminator yields better results
         and hence not conditioning the discriminator with the input image
      Authors also claim that using two discriminators for cVAE-GAN and cLR-GAN
         yields better results, here we share the weights for both of them

      Args:
         image      : Input image to the discriminator
         kernels    : Number of kernels for the first layer of the network
         num_layers : Total number of layers
         norm_layer : Type of normalization layer {batch/instance}
         non_lin    : Type of non-linearity of the network
         use_sigmoid: Use Sigmoid layer before the final layer?
         reuse      : Flag to check whether to reuse the variables created for the
                     discriminator graph
         norm       : Whether to use batch normalization layer

      Returns:
         Whether or not the input image is real or fake
      """
      self.d_layers = {}
      with tf.variable_scope('discriminator'):
         if not self.opts.d_usemulti:
            return self.discriminator_patch(image, kernels, num_layers, norm_layer, non_lin, 
               use_sigmoid, reuse, norm)
         else:
            raise NotImplementedError("Multiple discriminators is not implemented")

   def discriminator_patch(self, image, kernels, num_layers, norm_layer, non_lin,
                           use_sigmoid=False, reuse=False, norm=None):
      """PatchGAN discriminator
      """
      k, s = 4, 2
      self.d_layers['conv0'] = conv2d(image, ksize=k, out_channels=kernels*1, stride=s, name='conv0',
            non_lin=self.non_lin[non_lin], reuse=reuse)
      for idx in range(1, num_layers):
         input_layer = self.d_layers['conv{}'.format(idx-1)]
         factor = min(2**idx, 8)
         if not norm:
            self.d_layers['conv{}'.format(idx)] = conv2d(input_layer, ksize=k,
               out_channels=kernels*factor, stride=s, name='conv{}'.format(idx),
               non_lin=self.non_lin[non_lin], reuse=reuse)
         else:
            self.d_layers['conv{}'.format(idx)] = conv_bn_lrelu(input_layer, ksize=k,
               out_channels=kernels*factor, is_training=self.train_mode, stride=s,
               name='conv{}'.format(idx), reuse=reuse)

      input_layer = self.d_layers['conv{}'.format(num_layers-1)]
      factor = min(2**num_layers, 8)
      if not norm:
         self.d_layers['conv{}'.format(num_layers)] = conv2d(input_layer, ksize=k, out_channels=
            kernels*factor, stride=s, name='conv{}'.format(num_layers), reuse=reuse)
      else:
         self.d_layers['conv{}'.format(num_layers)] = conv_bn_lrelu(input_layer, ksize=k,
            out_channels=kernels*factor, is_training=self.train_mode, stride=s,
            name='conv{}'.format(num_layers), reuse=reuse)

      input_layer = self.d_layers['conv{}'.format(num_layers)]
      self.d_layers['conv{}'.format(num_layers+1)] = conv2d(input_layer, ksize=k, out_channels=1, 
         stride=s, name='conv{}'.format(num_layers+1), reuse=reuse)

      if use_sigmoid:
         return sigmoid(self.d_layers['conv{}'.format(num_layers+1)])

   def model_loss(self):
      """Implements the loss graph
      """

      def cVAE_GAN_loss():
         """Computes CVAE-GAN loss
         """
         self.loss['l1']  = l1_loss(self.target_images, self.G)
         self.loss['gan'] = gan_loss(self.D, self.D_)
         self.loss['kl']  = kl_divergence(self.E)

      def cLR_GAN_loss():
         """Computes CLR-GAN loss
         """
         self.loss['l1']  = l1_loss(self.E, self.code)
         self.loss['gan'] = gan_loss(self.D, self.D_)

      def gan_loss(true_logit, fake_logit):
         """Implements the GAN loss

         Args:
            true_logit: Output of discriminator for true image
            fake_logit: Output of discriminator for fake image

         Returns:

         """
         self.loss['G_fake_loss']
         self.loss['G_real_loss']
         self.loss['G_loss']
         self.loss['D_loss']
         self.loss['gan']

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

      def kl_divergence(p1, p2=None):
         """Apply KL divergence
         
         Args:
            p1: 1st probability distribution
            p2: 2nd probability distribution (Usually unit Gaussian distribution)

         Returns:
            KL Divergence between the given distributions
         """
         pass

      self.loss = {'l1': 0., 'kl': 0., 'gan': 0., 
                   'G_real_loss': .0,
                   'G_fake_loss': .0,
                   'G_loss': .0,
                   'D_loss': .0,
                   'enc': .0,
                   'gen_enc': .0
                  }
      if self.opts.model == 'cvae-gan':
         cVAE_GAN_loss()
      elif self.opts.model == 'clr-gan':
         cLR_GAN_loss()
      elif self.opts.model == 'bicycle':
         cVAE_GAN_loss()
         cLR_GAN_loss()
      else:
         raise ValueError("\"{}\" type of architecture doesn't exist for loss !".format(self.opts.model))

   def train(self):
      """Train the network
      """
      self.test_graph()
      self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
      data = Dataset()
      self.sess.run(self.init)
      formatter =  "{:s} Epoch: [{:4d}/{:4d}] Batch: [{:2d}/{:2d}]  LR: {:.5f} "
      formatter += "G_fake_loss: {:.5f} G_real_loss: {:.5f} G_loss: {:.5f} D_loss: {:.5f} "
      formatter += "E_loss: {:.5f}"
      for epoch in xrange(self.opts.max_epochs):
         batch_num = 0
         for batch_begin, batch_end in zip(xrange(0, data.train_size(),
            self.opts.batch_size), xrange(self.opts.batch_size, data.train_size(),
            self.opts.batch_size)):
            begin_time = datetime.now().strftime("%Y%m%d%H%M%S")
            itertion = epoch * (data.train_size()/self.opts.batch_size) + batch_num
            images_A, images_B = data.load_batch(batch_begin, batch_end)

            code = np.random.uniform(low=-1.0, high=1.0, 
                                     size=[self.opts.batch_size,
                                     self.opts.code_len]).astype(np.float32)
            feed_dict = {self.image_A: imagesA,
                         self.images_B: images_B,
                         self.code: code,
                         self.is_training: True
                        }

            # Fake Data
            _, fake_loss, real_loss, g_loss, d_loss, e_loss = self.sess.run(
               [self.D_opt, ],
               feed_dict=feed_dict)

            # Real Data
            _, fake_loss, real_loss, g_loss, d_loss, e_loss = self.sess.run(
               [self.GE_opt],
               feed_dict=feed_dict)

            if np.mod(iteration, self.opts.gen_frq) == 0:
               # Sample the images by setting is_training=False
               pass

            batch_num += 1

         if np.mod(epoch, self.opts.ckpt_frq) == 0:
            self.checkpoint(epoch)

   def checkpoint(self, epoch):
      """Creates a checkpoint at the given epoch

      Args:
         epoch: epoch number of the training process
      """
      self.saver.save(self.sess, os.path.join(self.opts.summary_dir, "model_{}.ckpt").format(epoch))

   def test_graph(self):
      """Generate the graph and check if the connections are correct
      """
      print 'Test graph is generated...'
      self.writer = tf.summary.FileWriter(self.opts.summary_dir, self.sess.graph)

   def test(self, image):
      """Test the model

      Args:
         image: Input image to the model

      Returns:
         The generated image conditioned on the input image
      """
      if image == '':
         raise ValueError('Specify the path to the test image')
      latest_ckpt = tf.train.latest_checkpoint(self.opts.ckpt)
      tf.saver.restore(self.sess, latest_ckpt)
