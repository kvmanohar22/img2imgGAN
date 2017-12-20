import numpy as np 
import tensorflow as tf


def weight_init(shape, name=None, initializer=tf.contrib.layers.xavier_initializer()):
   """
   Weights Initialization
   """
   if name is None:
      name='W'
   
   W = tf.get_variable(name=name, shape=shape,
      initializer=initializer)
   return W 


def bias_init(shape, name=None, constant=0.0):
   """
   Bias Initialization
   """
   if name is None:
      name='b'

   b = tf.get_variable(name=name, shape=shape,
      initializer=tf.constant_initializer(0.0))
   return b


def conv2d(input, kernel, out_channels, stride=1, name=None, reuse=False, 
   initializer=tf.contrib.layers.xavier_initializer(), bias_constant=0.01,
   non_lin=None):   
   """
   2D convolution layer with relu activation
   """
   if name is None:
      name='2d_convolution'

   in_channels = input.get_shape().as_list()[-1]
   kernel = [kernel, kernel, in_channels, out_channels]
   with tf.variable_scope(name, reuse=reuse):
      W = weight_init(kernel, 'W', initializer)
      b = bias_init(kernel[3], 'b', bias_constant)

      strides=[1, stride, stride, 1]
      output = tf.nn.conv2d(input=input, filter=W, strides=strides, padding='SAME')
      output = output + b
      if non_lin is None:
         return output
      else:
         return non_lin(output)

def deconv(input, kernel, out_shape, out_channels, stride=1, name=None,
   reuse=False, initializer=tf.contrib.layers.xavier_initializer(),
   bias_constant=0.0, batch_size=1, non_lin=None):
   """
   2D deconvolution layer with relu activation
   """
   if name is None:
      name='de_convolution'

   in_channels = input.get_shape().as_list()[-1]
   kernel = [kernel, kernel, out_channels, in_channels]
   output_shape = [batch_size, out_shape, out_shape, out_channels]
   with tf.variable_scope(name, reuse=reuse):
      W = weight_init(kernel, 'W', initializer)
      b = bias_init(kernel[2], 'b', bias_constant)

      strides=[1, stride, stride, 1]
      output = tf.nn.conv2d_transpose(value=input, filter=W, output_shape=output_shape, strides=strides)
      output = output + b
      if non_lin is None:
         return output
      else:
         return non_lin(output)

def max_pool(input, kernel=3, stride=2, name=None):
   """
   Max-pool
   """
   if name is None: 
      name='max_pool'

   with tf.variable_scope(name):
      ksize = [1, kernel, kernel, 1]
      strides = [1, stride, stride, 1]
      output = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')
      return output

def average_pool(input, ksize=3, strides=3, padding='VALID', name=None):
   """Average Pooling layer
   """
   if name is None:
      name='avg_pool'

   with tf.variable_scope(name):
      ksize = [1, ksize, ksize, 1]
      strides = [1, strides, strides, 1]
      output = tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding=padding)
      return output

def fully_connected(input, output_neurons, name=None, reuse=False,
   bias_constant=0.01, initializer=tf.contrib.layers.xavier_initializer()):
   """Fully-connected linear activations
   """
   if name is None:
      name='fully_connected'

   shape = input.get_shape()
   input_units = int(shape[1])
   with tf.variable_scope(name, reuse=reuse):
      W = weight_init([input_units, output_neurons], 'W', initializer)
      b = bias_init([output_neurons], 'b', bias_constant)

      output = tf.add(tf.matmul(input, W), b)
      return output


def dropout_layer(input, keep_prob=0.5, name=None):
   """Dropout layer
   """
   if name is None:
      name='Dropout'

   with tf.variable_scope(name):
      output = tf.nn.dropout(input, keep_prob=keep_prob)
      return output

def relu(input_layer, name=None):
   """
   ReLU activation
   """
   if name is None:
      name = "relu"

   with tf.variable_scope(name):
      return tf.nn.relu(input_layer)

def tanh(input_layer, name=None):
   """
   Tanh activation
   """
   if name is None:
      name = "tanh"

   with tf.variable_scope(name):
      return tf.nn.tanh(input_layer)

def sigmoid(input_layer, name=None):
   """
   Tanh activation
   """
   if name is None:
      name = "sigmoid"

   with tf.variable_scope(name):
      return tf.nn.sigmoid(input_layer)

def lrelu(input, alpha=0.2, name=None):
   """
   Leaky ReLU
   """
   if name is None:
      name = "lrelu"

   with tf.variable_scope(name):
      return tf.maximum(input, alpha * input)

def batch_normalize(input_layer, is_training, reuse=False, name=None):
   """
   Applies batch normalization
   """
   if name is None:
      name = "BatchNorm"

   with tf.variable_scope(name, reuse=reuse):
      output = tf.contrib.layers.batch_norm(
         input_layer,
         center=True,
         scale=True,
         is_training=is_training)
      return output


def conv_bn_relu(input_layer, kernel, out_channels, is_training, stride=1, name=None,
   reuse=False, initializer=tf.contrib.layers.xavier_initializer(), bias_constant=0.01):
   """
   Applies series of operations of `conv`->`batch_norm`->`relu`
   """
   with tf.variable_scope(name+'_block', reuse=reuse):
      conv_ac = conv2d(input_layer, kernel, out_channels, stride, name, reuse)
      conv_bn = batch_normalize(conv_ac, is_training, reuse=reuse)
      conv_rl = relu(conv_bn)
      return conv_rl


def conv_bn_lrelu(input_layer, kernel, out_channels, is_training, stride=1, name=None,
   reuse=False, initializer=tf.contrib.layers.xavier_initializer(), bias_constant=0.01):
   """
   Applies series of operations of `conv`->`batch_norm`->`leaky_relu`
   """
   with tf.variable_scope(name+'_block', reuse=reuse):
      conv_ac = conv2d(input_layer, kernel, out_channels, stride, name, reuse)
      conv_bn = batch_normalize(conv_ac, is_training, reuse=reuse)
      conv_rl = leaky_relu(conv_bn)
      return conv_rl


def dconv_bn_relu(input_layer, kernel, out_channels, out_shape, is_training, stride=1, name=None,
   reuse=False, initializer=tf.contrib.layers.xavier_initializer(), bias_constant=0.01, batch_size=1):
   """
   Applies series of operations of `conv`->`batch_norm`->`relu`
   """
   with tf.variable_scope(name+'_block', reuse=reuse):
      conv_ac = deconv(input_layer, kernel, out_shape, out_channels, stride, name, reuse, batch_size=batch_size)
      conv_bn = batch_normalize(conv_ac, is_training, reuse=reuse)
      conv_rl = relu(conv_bn)
      return conv_rl


def dconv_bn_lrelu(input_layer, kernel, out_channels, is_training, stride=1, name=None,
   reuse=False, initializer=tf.contrib.layers.xavier_initializer(), bias_constant=0.01, batch_size=1):
   """
   Applies series of operations of `conv`->`batch_norm`->`leaky_relu`
   """
   with tf.variable_scope(name+'_block', reuse=reuse):
      conv_ac = deconv(input_layer, kernel, out_channels, stride, name, reuse)
      conv_bn = batch_normalize(conv_ac, is_training, reuse=reuse)
      conv_rl = leaky_relu(conv_bn)
      return conv_rl


def residual_block(input_layer, output_channels, is_training, stride=1, first_block=False, name=None, reuse=False):
   """Builds a residual block
   Series of operations include : 
      -> bactch_norm
      -> relu
      -> conv
   """
   if name is None:
      name = "residual_block"

   input_channels = input_layer.get_shape().as_list()[-2]
   with tf.variable_scope(name, reuse=reuse):
      # First conv block 
      with tf.variable_scope("block_1"):
         conv1_bn = batch_normalize(input_layer, is_training, reuse=reuse)
         conv1_ac = relu(conv1_bn)
         conv1 = conv2d(conv1_ac, kernel=3, stride=stride, name="conv1", reuse=reuse)
      # Second conv block
      with tf.variable_scope("block_2"):
         conv2_bn = batch_normalize(conv1, is_training, reuse=reuse)
         conv2_ac = relu(conv2_bn)
         conv2 = conv2d(conv2_ac, kernel=3, stride=stride, name="conv2", reuse=reuse)
      if conv2.get_shape().as_list()[-1] != input_layer.get_shape().as_list()[-1]:
         raise ValueError('Output and input channels do not match')
      else:
         output = input_layer + conv2
      return output
   
def add_layers(layer_1, layer_2):
   """Adds two layers

   Args:
      layer_1: The first layer
      layer_2: The second layer

   Returns:
      Layer after addition of the given two
   """
   with tf.name_scope('addition'):
      return tf.add(layer_1, layer_2)

def activation_summary(tensor):
   """
   Write the summary of a tensor
   """
   tensor_name = tensor.op.name
   tf.summary.histogram(tensor_name+'/activation', tensor)
   tf.summary.scalar(tensor_name+'/sparsity', tf.nn.zero_fraction(tensor))
