"""Contains various graphs for building the entire model
   __author__ = "kvmanohar22"
"""

from datetime import datetime
import numpy as np
import os
import sys

from modules import *
from utils import Dataset
from utils import utils


class Model(object):
   """Defines the base class for all models
   """

   def __init__(self, opts, is_training):
      """Initialize the model by creating various parts of the graph

      Args:
         opts: All the hyper-parameters of the network
      """
      self.opts = opts
      self.h = opts.h
      self.w = opts.w
      self.c = opts.c
      self.train_mode = is_training
      self.sess = tf.Session()
      self.build_graph()

   def build_graph(self):
      """Generate various parts of the graph
      """
      sys.stdout.write(' - Building various parts of the graph...\n')
      self.non_lin = {'relu' : lambda x: relu(x, name='relu'),
                      'lrelu': lambda x: lrelu(x, name='lrelu'),
                      'tanh' : lambda x: tanh(x, name='tanh')
                      }
      self.allocate_placeholders()

      # Common discriminator
      self.D, self.D_logits = self.discriminator(self.target_images, self.opts.d_kernels,
                                                 self.opts.d_layers, non_lin=self.opts.d_nonlin,
                                                 norm=self.opts.d_norm, use_sigmoid=self.opts.d_sigmoid,
                                                 reuse=False)

      # Generators and Encoders
      if self.opts.model == 'cvae-gan':
         self.E_mean, self.E_std  = self.encoder(self.target_images, self.opts.e_layers,
                                                 self.opts.e_kernels, self.opts.e_nonlin,
                                                 norm=self.opts.e_norm, reuse=False)
         self.assign_gen_code()
         self.G  = self.generator(self.input_images, self.gen_input_noise, self.opts.g_layers,
                                  self.opts.g_kernels, self.opts.g_nonlin,
                                  norm=self.opts.g_norm)
      elif self.opts.model == 'clr-gan':
         self.assign_gen_code()
         self.G  = self.generator(self.input_images, self.gen_input_noise, self.opts.g_layers,
                                  self.opts.g_kernels, self.opts.g_nonlin,
                                  norm=self.opts.g_norm)
         self.E_mean, self.E_std  = self.encoder(self.G, self.opts.e_layers,
                                                 self.opts.e_kernels, self.opts.e_nonlin,
                                                 norm=self.opts.e_norm, reuse=False)
      elif self.opts.model == 'bicycle':
         # cVAE-GAN graph
         self.E_mean_1, self.E_std_1  = self.encoder(self.target_images, self.opts.e_layers,
                                                     self.opts.e_kernels, self.opts.e_nonlin,
                                                     norm=self.opts.e_norm, reuse=False)
         with tf.variable_scope('encoded_noise'):
            self.encoded_noise = self.E_mean_1 + self.code * self.E_std_1
         self.G_cvae = self.generator(self.input_images, self.encoded_noise, self.opts.g_layers,
                                      self.opts.g_kernels, self.opts.g_nonlin,
                                      norm=self.opts.g_norm)

         # cLR-GAN graph
         self.G_clr = self.generator(self.input_images, self.code, self.opts.g_layers,
                                     self.opts.g_kernels, self.opts.g_nonlin,
                                     norm=self.opts.g_norm, reuse=True)
         self.E_mean_2, self.E_std_2  = self.encoder(self.G_clr, self.opts.e_layers,
                                                     self.opts.e_kernels, self.opts.e_nonlin,
                                                     norm=self.opts.e_norm, reuse=True)

         # Discriminators
         self.D_cvae, self.D_cvae_logits_ = self.discriminator(self.G_cvae, self.opts.d_kernels,
                                                 self.opts.d_layers, non_lin=self.opts.d_nonlin,
                                                 norm=self.opts.d_norm, use_sigmoid=self.opts.d_sigmoid,
                                                 reuse=True)
         self.D_clr, self.D_clr_logits_   = self.discriminator(self.G_clr, self.opts.d_kernels,
                                                 self.opts.d_layers, non_lin=self.opts.d_nonlin,
                                                 norm=self.opts.d_norm, use_sigmoid=self.opts.d_sigmoid,
                                                 reuse=True)

      if self.opts.model == 'cvae-gan' or self.opts.model == 'clr-gan':
         self.D_, self.D_logits_ = self.discriminator(self.G, self.opts.d_kernels, self.opts.d_layers,
                                                      non_lin=self.opts.d_nonlin, norm=self.opts.d_norm,
                                                      use_sigmoid=self.opts.d_sigmoid,
                                                      reuse=True)

      self.variables = tf.trainable_variables()
      self.d_vars = [var for var in self.variables if 'discriminator' in var.name]
      self.ge_vars = [var for var in self.variables if 'generator' or 'encoder' in var.name]
      self.model_loss()
      self.D_opt = tf.train.AdamOptimizer(self.opts.base_lr).minimize(self.d_loss, var_list=self.d_vars)
      self.GE_opt = tf.train.AdamOptimizer(self.opts.base_lr).minimize(self.g_loss, var_list=self.ge_vars)
      self.summaries()

   def allocate_placeholders(self):
      """Allocate placeholders of the graph
      """
      sys.stdout.write(' - Allocating placholders...\n')
      self.images_A = tf.placeholder(tf.float32, [None, self.h, self.w, self.c], name="images_A")
      self.images_B = tf.placeholder(tf.float32, [None, self.h, self.w, self.c], name="images_B")
      self.code = tf.placeholder(tf.float32, [None, self.opts.code_len], name="code")
      self.is_training = tf.placeholder(tf.bool, name='is_training')
      if self.opts.direction == 'a2b':
         self.input_images  = self.images_A
         self.target_images = self.images_B
      elif self.opts.direction == 'b2a':
         self.input_images  = self.images_B
         self.target_images = self.images_A
      else:
         raise ValueError("There is no such image transition type")

   def assign_gen_code(self):
      """Assigns the noise for the generator
         This is dynamic: during Train mode, noise is the encoded vector
      """
      def train_mode():
         """Noise to be set during training mode"""
         input_noise = None
         if self.opts.model == 'cvae-gan' or self.opts.model == 'bicycle':
            input_noise = self.E_mean + self.code * self.E_std
         elif self.opts.model == 'clr-gan':
            input_noise = self.code
         else:
            raise ValueError("No such type of model exists !")
         return input_noise

      def test_mode():
         """Noise to be set during test mode"""
         return self.code

      with tf.variable_scope('Noise'):
         self.gen_input_noise = tf.cond(tf.equal(self.is_training, tf.constant(True)),
                                        true_fn=train_mode,
                                        false_fn=test_mode,
                                        name='Noise')
         assert self.gen_input_noise is not None, "Generator input noise is not fed"

   def summaries(self):
      """Adds all the necessary summaries
      """
      images_A = tf.summary.image('images_A', self.images_A, max_outputs=10)
      images_B = tf.summary.image('images_B', self.images_B, max_outputs=10)
      if self.opts.model == 'bicycle':
         gen_images_cvae = tf.summary.image('Gen_images_cVAE', self.G_cvae, max_outputs=10)
         gen_images_clr  = tf.summary.image('Gen_images_cLR', self.G_clr, max_outputs=10)
         gen_images = [gen_images_clr, gen_images_cvae]
      else:
         gen_images = tf.summary.image('Gen_images', self.G, max_outputs=10)

      # Just to visualize how images are generated
      self.gen_images_cvae_sample = tf.summary.image('Gen_images_cVAE_sample', self.G_cvae, max_outputs=10)
      self.gen_images_clr_sample  = tf.summary.image('Gen_images_cLR_sample', self.G_clr, max_outputs=10)

      # Loss
      z_summary = tf.summary.histogram('z', self.code)
      d_loss_fake = tf.summary.scalar('D_loss_fake', self.loss['D_fake_loss'])
      d_loss_real = tf.summary.scalar('D_loss_real', self.loss['D_real_loss'])
      d_loss = tf.summary.scalar('D_loss', self.d_loss)
      g_loss = tf.summary.scalar('G_loss', self.g_loss)
      self.d_summaries = tf.summary.merge([d_loss_fake, d_loss_real, z_summary, d_loss])
      if self.opts.model == 'bicycle':
         self.g_summaries = tf.summary.merge([g_loss])#, images_A, images_B]+gen_images)
      else:
         self.g_summaries = tf.summary.merge([g_loss, images_A, images_B]+[gen_images])
      if not self.opts.full_summaries:
         self.act_sparsity = tf.summary.merge(tf.get_collection('hist_spar'))
      self.debug_summaries = tf.summary.merge([self.gen_images_clr_sample, self.gen_images_cvae_sample])

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
            non_lin=self.non_lin[non_lin], reuse=reuse)
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
               out_channels=kernels*factor, is_training=self.is_training, stride=s,
               name='conv{}'.format(idx), reuse=reuse)
         if not self.opts.full_summaries:
            activation_summary(self.e_layers['conv{}'.format(idx)])

      self.e_layers['pool'] = average_pool(self.e_layers['conv{}'.format(num_layers-1)],
         ksize=8, stride=8, name='pool')
      if not self.opts.full_summaries:
         activation_summary(self.e_layers['pool'])

      units = int(np.prod(self.e_layers['pool'].get_shape().as_list()[1:]))
      reshape_layer = tf.reshape(self.e_layers['pool'], [-1, units])
      self.e_layers['full_mean'] = fully_connected(reshape_layer, output_neurons, name='full_mean',
                                                   reuse=reuse)
      # This layers predicts the `log(var)`, to get the std,
      # std = exp(0.5 * log(var))
      self.e_layers['full_logvar'] = fully_connected(reshape_layer, output_neurons, name='full_logvar',
                                                  reuse=reuse)
      if not self.opts.full_summaries:
         activation_summary(self.e_layers['full_mean'])
         activation_summary(self.e_layers['full_logvar'])

      return self.e_layers['full_mean'], tf.exp(0.5 * self.e_layers['full_logvar'])

   def resnet_encoder(self, image, num_layers=4, output_neurons=1, kernels=64, non_lin='relu',
                      norm=None, reuse=False):
      """Residual Network with several residual blocks
      """
      raise NotImplementedError("Not Implemented")

   def generator(self, image, z, layers=3, kernels=64, non_lin='relu', norm=None,
                 reuse=False):
      """Generator graph of GAN

      Args:
         image  : Conditioned image on which the generator generates the image 
         z      : Latent space code (or noise when sampling the images)
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

   def generator_input(self, image, z, layers=3, kernels=32, non_lin='lrelu', norm=None,
                       reuse=False):
      """Generator graph where noise is concatenated to the first layer
      """

      with tf.name_scope('replication'):
         tiled_z = tf.tile(z, [1, self.w*self.h], name='tiling')
         reshaped = tf.reshape(tiled_z, [-1, self.h, self.w, self.opts.code_len], name='reshape')
         in_layer = tf.concat([image, reshaped], axis=3, name='concat')
      k, s = 4, 2
      factor = 1

      with tf.variable_scope('down_1'):
        conv1 = conv2d(in_layer, ksize=3, out_channels=32, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv1 = conv2d(conv1,    ksize=3, out_channels=32, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
        pool1 = max_pool(conv1, kernel=2, stride=2, name='pool1') #128
      
      with tf.variable_scope('down_2'):
        conv2 = conv2d(pool1, ksize=3, out_channels=64, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv2 = conv2d(conv2, ksize=3, out_channels=64, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
        pool2 = max_pool(conv2, kernel=2, stride=2, name='pool1') #64

      with tf.variable_scope('down_3'):
        conv3 = conv2d(pool2, ksize=3, out_channels=128, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv3 = conv2d(conv3, ksize=3, out_channels=128, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
        pool3 = max_pool(conv3, kernel=2, stride=2, name='pool1') #32

      with tf.variable_scope('down_4'):
        conv4 = conv2d(pool3, ksize=3, out_channels=256, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv4 = conv2d(conv4, ksize=3, out_channels=256, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
        pool4 = max_pool(conv4, kernel=2, stride=2, name='pool1') #16

      with tf.variable_scope('down_5'):
        conv5  = conv2d(pool4, ksize=3, out_channels=512, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv5 = conv2d(conv5,  ksize=3, out_channels=512, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)


      with tf.variable_scope('up_1'):
        dcnv1 = deconv(conv5, ksize=3, out_channels=512, stride=2, name='dconv1', out_shape=32, non_lin=self.non_lin[non_lin],
                       batch_size=self.opts.batch_size, reuse=reuse)
        # h, w = conv4.get_shape().as_list()[1], conv4.get_shape().as_list()[2]
        # sliced = tf.slice(conv4, [0, h//2, w//2, 0], [0, h//2, w//2, 0], name='slice1')
        up1   = concatenate(dcnv1, conv4, axis=3)
        conv6 = conv2d(up1,   ksize=3, out_channels=256, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv6 = conv2d(conv6, ksize=3, out_channels=256, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

      with tf.variable_scope('up_2'):
        dcnv2 = deconv(conv6, ksize=3, out_channels=256, stride=2, name='dconv1', out_shape=64, non_lin=self.non_lin[non_lin],
                       batch_size=self.opts.batch_size, reuse=reuse)
        # h, w = conv3.get_shape().as_list()[1], conv3.get_shape().as_list()[2]
        # sliced = tf.slice(conv3, [0, h//2, w//2, 0], [0, h//2, w//2, 0], name='slice1')
        up2   = concatenate(dcnv2, conv3, axis=3)
        conv7 = conv2d(up2,   ksize=3, out_channels=128, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv7 = conv2d(conv7, ksize=3, out_channels=128, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

      with tf.variable_scope('up_3'):
        dcnv3 = deconv(conv7, ksize=3, out_channels=128, stride=2, name='dconv1', out_shape=128, non_lin=self.non_lin[non_lin],
                       batch_size=self.opts.batch_size, reuse=reuse)
        # h, w = conv2.get_shape().as_list()[1], conv2.get_shape().as_list()[2]
        # sliced = tf.slice(conv2, [0, h//2, w//2, 0], [0, h//2, w//2, 0], name='slice1')
        up2   = concatenate(dcnv3, conv2, axis=3)
        conv8 = conv2d(up2,   ksize=3, out_channels=64, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv8 = conv2d(conv8, ksize=3, out_channels=64, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

      with tf.variable_scope('up_4'):
        dcnv4 = deconv(conv8, ksize=3, out_channels=64, stride=2, name='dconv1', out_shape=256, non_lin=self.non_lin[non_lin],
                       batch_size=self.opts.batch_size, reuse=reuse)
        # h, w = conv1.get_shape().as_list()[1], conv1.get_shape().as_list()[2]
        # sliced = tf.slice(conv1, [0, h//2, w//2, 0], [0, h//2, w//2, 0], name='slice1')
        up3   = concatenate(dcnv4, conv1, axis=3)
        conv9 = conv2d(up3,   ksize=3, out_channels=32, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
        conv9 = conv2d(conv9, ksize=3, out_channels=32, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

      with tf.variable_scope('up_5'):
        output = conv2d(conv9, ksize=3, out_channels=3, stride=1, name='conv1', non_lin=self.non_lin['tanh'], reuse=reuse)

      return output
      # # Downsampling
      # for idx in xrange(layers):
      #    factor = min(2**idx, 4)
      #    if not norm:
      #       self.g_layers['conv{}'.format(idx)] = conv2d(in_layer, ksize=k, 
      #          out_channels=kernels*factor, stride=s, name='conv{}'.format(idx),
      #          non_lin=self.non_lin[non_lin], reuse=reuse)
      #    else:
      #       self.g_layers['conv{}'.format(idx)] = conv_bn_relu(in_layer, ksize=k, 
      #          out_channels=kernels*factor, is_training=self.is_training, stride=s,
      #          name='conv{}'.format(idx), reuse=reuse)
      #    if not self.opts.full_summaries:
      #       activation_summary(self.g_layers['conv{}'.format(idx)])
      #    in_layer = self.g_layers['conv{}'.format(idx)]

      # # Upsampling
      # in_layer = self.g_layers['conv{}'.format(layers-1)]
      # new_idx = layers
      # for idx in xrange(layers-2, -1, -1):
      #    input_shape = self.g_layers['conv{}'.format(idx)].get_shape().as_list()
      #    out_shape = input_shape[1]
      #    out_channels = input_shape[-1]
      #    self.g_layers['deconv{}'.format(new_idx)] = deconv(in_layer, ksize=k,
      #       out_shape=out_shape, out_channels=out_channels, name='deconv{}'.format(new_idx),
      #       non_lin=self.non_lin[non_lin], batch_size=self.opts.batch_size, stride=s, reuse=reuse)
      #    if not self.opts.full_summaries:
      #       activation_summary(self.g_layers['deconv{}'.format(new_idx)])

      #    input_layer = self.g_layers['deconv{}'.format(new_idx)]
      #    self.g_layers['deconv{}_add'.format(new_idx)] = add_layers(input_layer,
      #       self.g_layers['conv{}'.format(idx)])
      #    in_layer = self.g_layers['deconv{}_add'.format(new_idx)]
      #    new_idx += 1

      # self.g_layers['deconv{}'.format(layers*2-1)] = deconv(self.g_layers['deconv{}_add'.format(new_idx-1)],
      #    ksize=k, out_shape=self.h, out_channels=3, name='deconv{}'.format(new_idx),
      #    non_lin=self.non_lin['tanh'], batch_size=self.opts.batch_size, stride=s, reuse=reuse)
      # if not self.opts.full_summaries:
      #    activation_summary(self.g_layers['deconv{}'.format(layers*2-1)])

      # return self.g_layers['deconv{}'.format(layers*2-1)]


   def generator_all(self, image, z, layers=3, kernels=64, non_lin='lrelu', norm=None,
                     reuse=False):
      """Generator graph where noise is to all the layers
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
               out_channels=kernels*factor, is_training=self.is_training, stride=s,
               name='conv{}'.format(idx), reuse=reuse)

      input_layer = self.d_layers['conv{}'.format(num_layers-1)]
      factor = min(2**num_layers, 8)
      if not norm:
         self.d_layers['conv{}'.format(num_layers)] = conv2d(input_layer, ksize=k, out_channels=
            kernels*factor, stride=s, name='conv{}'.format(num_layers), non_lin=self.non_lin[non_lin],
            reuse=reuse)
      else:
         self.d_layers['conv{}'.format(num_layers)] = conv_bn_lrelu(input_layer, ksize=k,
            out_channels=kernels*factor, is_training=self.is_training, stride=s,
            name='conv{}'.format(num_layers), reuse=reuse)

      input_layer = self.d_layers['conv{}'.format(num_layers)]
      self.d_layers['conv{}'.format(num_layers+1)] = conv2d(input_layer, ksize=k, out_channels=1, 
         stride=s, name='conv{}'.format(num_layers+1), reuse=reuse)

      logits = self.d_layers['conv{}'.format(num_layers+1)]
      return sigmoid(logits), logits

   def model_loss(self):
      """Implements the loss graph
         All the loss values are stored in the dictionary `self.loss`
      """

      def cVAE_GAN_loss(true_logit, fake_logit, E_mean, E_std, z1, z2):
         """Computes cVAE-GAN loss

         Args:
            true_logit: Output of discriminator for true image
            fake_logit: Output of discriminator for fake image
            E_mean    : Mean predicted by encoder
            E_std     : Std predicted by encoder
            z1        : -
            z2        : -
         """
         with tf.variable_scope('cVAE_GAN_loss'):
            gan_loss(true_logit=true_logit,
                     fake_logit=fake_logit,
                     model='cVAE')
            with tf.variable_scope('KL_loss'):
               self.loss['KL']  = self.opts.lambda_kl * kl_divergence(E_mean, E_std)
            with tf.variable_scope('L1_VAE_loss'):
               self.loss['L1_VAE']  = self.opts.lambda_img * l1_loss(z1, z2)

      def cLR_GAN_loss(true_logit, fake_logit, E, skip=False):
         """Computes cLR-GAN loss

         Args:
            true_logit: Output of discriminator for true image
            fake_logit: Output of discriminator for fake image
            E         : Mean predicted by encoder
            skip      : Whether to skip the G_loss
         """
         with tf.variable_scope('cLR_GAN_loss'):
            gan_loss(true_logit=true_logit,
                     fake_logit=fake_logit,
                     model='cLR',
                     skip_d_real_loss=skip)
            with tf.variable_scope('L1_latent_loss'):
               self.loss['L1_latent']  = self.opts.lambda_latent * l1_loss(E, self.code)

      def gan_loss(true_logit, fake_logit, model='cLR', skip_d_real_loss=False):
         """Implements the GAN loss

         Args:
            true_logit      : Output of discriminator for true image
            fake_logit      : Output of discriminator for fake image
            model           : Name of the model to compute loss for
            skip_d_real_loss: Whether to skip G_loss, should be skipped the second time
                              while training bicycleGAN model
         """
         if len(true_logit.get_shape().as_list()) != 2:
            true_logit = tf.reduce_mean(tf.reshape(true_logit, [self.opts.batch_size, -1]), axis=1)
            fake_logit = tf.reduce_mean(tf.reshape(fake_logit, [self.opts.batch_size, -1]), axis=1)
         with tf.variable_scope('D_fake_loss'):
            self.loss['D_{}_fake_loss'.format(model)] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=fake_logit, labels=tf.zeros_like(fake_logit)))
         with tf.variable_scope('D_real_loss'):
            if not skip_d_real_loss:
               self.loss['D_{}_real_loss'.format(model)] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                     logits=true_logit, labels=tf.ones_like(true_logit)))
            else:
               self.loss['D_{}_real_loss'.format(model)] = 0.
         with tf.variable_scope('G_loss'):
            self.loss['G_{}_loss'.format(model)] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=fake_logit, labels=tf.ones_like(fake_logit)))

         self.loss['D_{}_loss'.format(model)] = self.loss['D_{}_fake_loss'.format(model)] + \
                                                self.loss['D_{}_real_loss'.format(model)]

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
         return tf.reduce_mean(tf.abs(z1-z2))

      def kl_divergence(p1_mean, p1_std):
         """Apply KL divergence
            The second distribution is assumed to be unit Gaussian distribution
         
         Args:
            p1_mean: Mean of 1st probability distribution
            p1_std : Std of 1st probability distribution

         Returns:
            KL Divergence between the given distributions
         """
         divergence = 0.5 * tf.reduce_sum(tf.square(p1_mean)+tf.square(p1_std)- \
                                          1.0 * tf.log(tf.square(p1_std))-1, axis=1)
         return tf.reduce_mean(divergence, axis=0)

      with tf.variable_scope('loss'):
         self.loss = {}
         if self.opts.model == 'cvae-gan':
            cVAE_GAN_loss(self.D_logits, self.D_logits_, self.E_mean,
                          self.E_std, self.target_images, self.G)
            self.d_loss = self.loss['D_cVAE_loss']
            self.g_loss = self.loss['KL'] +\
                          self.loss['L1_VAE'] +\
                          self.loss['G_cVAE_loss']
            self.loss['D_fake_loss'] = self.loss['D_cVAE_fake_loss']
            self.loss['D_real_loss'] = self.loss['D_cVAE_real_loss']
         elif self.opts.model == 'clr-gan':
            cLR_GAN_loss(self.D_logits, self.D_logits_, self.E_mean)
            self.d_loss = self.loss['D_cLR_loss']
            self.g_loss = self.loss['L1_latent'] +\
                          self.loss['G_cLR_loss']
            self.loss['D_fake_loss'] = self.loss['D_cLR_fake_loss']
            self.loss['D_real_loss'] = self.loss['D_cLR_real_loss']
         elif self.opts.model == 'bicycle':
            with tf.variable_scope('Bicycle_GAN_loss'):
               cVAE_GAN_loss(self.D_logits, self.D_cvae_logits_, self.E_mean_1,
                             self.E_std_1, self.target_images, self.G_cvae)
               cLR_GAN_loss(self.D_logits, self.D_clr_logits_, self.E_mean_2, skip=True)
               self.d_loss = self.loss['D_cLR_loss'] +\
                             self.loss['D_cVAE_loss']
               self.g_loss = self.loss['KL'] +\
                             self.loss['L1_VAE'] +\
                             self.loss['L1_latent'] + \
                             self.loss['G_cLR_loss'] +\
                             self.loss['G_cVAE_loss']
               self.loss['D_fake_loss'] = self.loss['D_cLR_fake_loss'] +\
                                          self.loss['D_cVAE_fake_loss']
               self.loss['D_real_loss'] = self.loss['D_cVAE_real_loss']
         else:
            raise ValueError("\"{}\" type of architecture doesn't exist for loss !".format(self.opts.model))

   def train(self):
      """Train the network
      """
      self.test_graph()
      self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
      data = Dataset(self.opts, load=True)
      self.init = tf.global_variables_initializer()
      self.sess.run(self.init)
      formatter =  "Elapsed Time: {}  Epoch: [{:3d}/{:4d}]  Batch: [{:3d}/{:3d}]  LR: {:.5f}  "
      formatter += "D_fake_loss: {:.5f}  D_real_loss: {:.5f}  D_loss: {:.5f}  G_loss: {:.5f}"

      runtime_z = np.random.uniform(low=-1,
                                    high=1,
                                    size=[self.opts.sample_num,
                                          self.opts.code_len]).astype(np.float32)
      start_time = datetime.now()
      print ' - Training the network...\n'
      for epoch in xrange(self.opts.max_epochs):
         batch_num = 0
         for batch_begin, batch_end in zip(xrange(0, data.train_size(),
            self.opts.batch_size), xrange(self.opts.batch_size, data.train_size()+1,
            self.opts.batch_size)):

            iteration = epoch * (data.train_size()/self.opts.batch_size) + batch_num
            images_A, images_B = data.load_batch(batch_begin, batch_end)

            code = np.random.uniform(low=-1.0,
                                     high=1.0,
                                     size=[self.opts.batch_size,
                                           self.opts.code_len]).astype(np.float32)

            # Update Discriminator
            feed_dict = {self.images_A: images_A,
                         self.images_B: images_B,
                         self.code: code,
                         self.is_training: True
                        }
            _, d_loss, d_summaries, d_fake, d_real = self.sess.run(
                    [self.D_opt, self.d_loss, self.d_summaries,
                     self.loss['D_fake_loss'],
                     self.loss['D_real_loss']
                     ],
                    feed_dict=feed_dict)
            # _, d_summaries, d_loss, d_fake, d_real = self.sess.run(
            #         [self.D_opt, self.d_summaries, self.d_loss,
            #          self.loss['D_fake_loss'],
            #          self.loss['D_real_loss']
            #          ],
            #         feed_dict=feed_dict)
            self.writer.add_summary(d_summaries, iteration)

            # Update Generator and Encoder
            feed_dict = {self.images_A: images_A,
                         self.images_B: images_B,
                         self.code: code,
                         self.is_training: True
                        }
            _, g_loss, g_summaries = self.sess.run(
                    [self.GE_opt, self.g_loss, self.d_summaries],
                    feed_dict=feed_dict)
            self.writer.add_summary(g_summaries, iteration)
            # self.writer.add_summary(sparsity_sum, iteration)
            # _, g_summaries, sparsity_sum, g_loss = self.sess.run(
            #         [self.GE_opt, self.g_summaries, self.act_sparsity,
            #          self.g_loss], feed_dict=feed_dict)
            # self.writer.add_summary(g_summaries, iteration)
            # self.writer.add_summary(sparsity_sum, iteration)

            elapsed_time = datetime.now() - start_time
            print formatter.format(elapsed_time, epoch, self.opts.max_epochs,
                                   batch_num+1, data.train_size()/self.opts.batch_size,
                                   self.opts.base_lr, d_fake, d_real, d_fake + d_real,
                                   g_loss)

            # Sample the images
            if np.mod(iteration, self.opts.gen_frq) == 0 and iteration != 0:
               print '[Sampling the images...]'
               feed_dict = {self.images_A: images_A,
                            self.images_B: images_B,
                            self.code: runtime_z,
                            self.is_training: False
                            }
               if self.opts.model == 'bicycle':
                  # images_cvae = self.G_cvae.eval(session=self.sess, feed_dict=feed_dict)
                  images_clr  = self.G_clr.eval(session=self.sess, feed_dict=feed_dict)
                  utils.imwrite(os.path.join(
                          self.opts.sample_dir, 'iter_{}_cLR'.format(iteration)),
                          images_clr, inv_normalize=True)
                  # utils.imwrite(os.path.join(
                  #         self.opts.sample_dir, 'iter_{}_cVAE'.format(iteration)),
                  #         images_cvae, inv_normalize=True)

                  # debug_summaries = self.debug_summaries.eval(session=self.sess,
                  #                                             feed_dict=feed_dict)
                  # self.writer.add_summary(debug_summaries, iteration)

               else:
                  images = self.G.eval(session=self.sess, feed_dict=feed_dict)
                  utils.imwrite(os.path.join(
                          self.opts.sample_dir, 'iter_{}'.format(iteration)),
                          images, inv_normalize=True)

            batch_num += 1

         if np.mod(epoch, self.opts.ckpt_frq) == 0:
            self.checkpoint(epoch)
      self.sess.close()

   def checkpoint(self, epoch):
      """Creates a checkpoint at the given epoch

      Args:
         epoch: epoch number of the training process
      """
      self.saver.save(self.sess, os.path.join(self.opts.summary_dir, "model_{}.ckpt").format(epoch))

   def test_graph(self):
      """Generate the graph and check if the connections are correct
      """
      sys.stdout.write(' - Generating the test graph...\n')
      self.writer = tf.summary.FileWriter(logdir=self.opts.summary_dir,
                                          graph=self.sess.graph)

   def validate(self, iteration):
      """Method to validate the model by sampling images at test time

      Args:
         iteration: Iteration number during training mode
      """

   def test(self, source):
      """Test the model

      Args:
         source: Input to the model, either single image or directory containing images

      Returns:
         The generated image conditioned on the input image
      """
      try:
         img = utils.imread(source)
      except IOError:
         image_paths = os.listdir(source)

      latest_ckpt = tf.train.latest_checkpoint(self.opts.ckpt_dir)
      self.saver.restore(self.sess, latest_ckpt)

      # TODO: Complete the forward pass and saving the images
