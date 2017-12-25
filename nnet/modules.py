import numpy as np 
import tensorflow as tf


def weight_init(shape, name=None,
                initializer=tf.contrib.layers.xavier_initializer()):
   """Weights Initialization

   Args:
      shape      : Shape of the variable
      name       : Name of the variable
      initializer: Type of weight initializer

   Returns:
      Initialized weight tensor
   """
   if name is None:
      name='W'

   W = tf.get_variable(name=name, shape=shape,
      initializer=initializer)
   return W 


def bias_init(shape, name=None, constant=0.0):
   """Bias Initialization

   Args:
      shape   : Shape of the variable
      name    : Name of the variable
      constant: Value of constant to initialize

   Returns:
      Initialized bias tensor
   """
   if name is None:
      name='b'

   b = tf.get_variable(name=name, shape=shape,
      initializer=tf.constant_initializer(constant))
   return b


def conv2d(input, ksize, out_channels, stride=1, name=None,
           reuse=False, 
           initializer=tf.contrib.layers.xavier_initializer(),
           bias_constant=0.01, non_lin=None):   
   """2D convolution layer with relu activation

   Args:
      input        : Input Tensor
      ksize        : filter's width (= filter's height)
      out_channels : Number of filters
      stride       : stride of the filter
      name         : Optional name for the operation
      reuse        : Whether to reuse this conv layer's weights and biases
      initializer  : Type of weight matrix initializer
      bias_constant: Constant value of bias
      non_lin      : Type of non-linearity for this layer

   Returns:
      Tensor after convolution operation
   """
   if name is None:
      name='2d_convolution'

   in_channels = input.get_shape().as_list()[-1]
   ksize = [ksize, ksize, in_channels, out_channels]
   with tf.variable_scope(name, reuse=reuse):
      W = weight_init(ksize, 'W', initializer)
      b = bias_init(ksize[3], 'b', bias_constant)

      strides=[1, stride, stride, 1]
      output = tf.nn.conv2d(input=input, filter=W, strides=strides, padding='SAME')
      output = output + b
      if non_lin is None:
         return output
      else:
         return non_lin(output)

def deconv(input, ksize, out_shape, out_channels, batch_size,
           stride=1, name=None, reuse=False, 
           initializer=tf.contrib.layers.xavier_initializer(),
           bias_constant=0.0, non_lin=None):
   """2D deconvolution layer with relu activation

   Args:
      input        : Input Tensor
      ksize        : filter's width (= filter's height)
      out_shape    : Shape of the output tensor
      out_channels : Number of filters
      batch_size   : Batch size
      stride       : stride of the filter
      name         : Optional name for the operation
      reuse        : Whether to reuse this conv layer's weights and biases
      initializer  : Type of weight matrix initializer
      bias_constant: Constant value of bias
      non_lin      : Type of non-linearity for this layer

   Returns:
      Tensor after convolution operation
   """
   if name is None:
      name='de_convolution'

   in_channels = input.get_shape().as_list()[-1]
   ksize = [ksize, ksize, out_channels, in_channels]
   output_shape = [batch_size, out_shape, out_shape, out_channels]
   with tf.variable_scope(name, reuse=reuse):
      W = weight_init(ksize, 'W', initializer)
      b = bias_init(ksize[2], 'b', bias_constant)

      strides=[1, stride, stride, 1]
      output = tf.nn.conv2d_transpose(value=input, filter=W,
         output_shape=output_shape, strides=strides)
      output = output + b
      if non_lin is None:
         return output
      else:
         return non_lin(output)

def max_pool(input, kernel=3, stride=2, name=None):
   """Max-pool

   Args:
      input : Input Tensor
      kernel: filter's width (= filter's height)
      stride: stride of the filter
      name  : Optional name for the operation

   Returns:
      Tensor after max-pool operation
   """
   if name is None: 
      name='max_pool'

   with tf.variable_scope(name):
      ksize = [1, kernel, kernel, 1]
      strides = [1, stride, stride, 1]
      output = tf.nn.max_pool(input, ksize=ksize, strides=strides,
         padding='SAME')
      return output

def average_pool(input, ksize=3, strides=3, padding='VALID', name=None):
   """Average Pooling layer

   Args:
      input : Input Tensor
      ksize : filter's width (= filter's height)
      stride: stride of the filter
      name  : Optional name for the operation

   Returns:
      Tensor after avg-pool operation
   """
   if name is None:
      name='avg_pool'

   with tf.variable_scope(name):
      ksize = [1, ksize, ksize, 1]
      strides = [1, strides, strides, 1]
      output = tf.nn.avg_pool(input, ksize=ksize, strides=strides,
         padding=padding)
      return output

def fully_connected(input, output_neurons, name=None, reuse=False,
                    bias_constant=0.01,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    non_lin=None):
   """Fully-connected linear activations

   Args:
      input         : Input Tensor
      output_neurons: filter's width (= filter's height)
      name          : Optional name for the operation
      reuse         : Whether to reuse this conv layer's weights and biases
      bias_constant : Constant value of bias
      initializer   : Type of weight matrix initializer
      non_lin       : Type of non-linearity for this layer

   Returns:
      Tensor after matrix multiplication operation
   """
   if name is None:
      name='fully_connected'

   shape = input.get_shape()
   input_units = int(shape[1])
   with tf.variable_scope(name, reuse=reuse):
      W = weight_init([input_units, output_neurons], 'W', initializer)
      b = bias_init([output_neurons], 'b', bias_constant)

      output = tf.add(tf.matmul(input, W), b)
      if non_lin is None:
         return output
      else:
         return non_lin(output)


def dropout_layer(input, keep_prob=0.5, name=None):
   """Dropout layer

   Args:
      input    : Input Tensor
      keep_prob: Keep Probability for each node
      name     : Optional name for the operation

   Returns:
      Tensor after dropout operation
   """
   if name is None:
      name='Dropout'

   with tf.variable_scope(name):
      output = tf.nn.dropout(input, keep_prob=keep_prob)
      return output

def relu(input, name=None):
   """ReLU activation

   Args:
      input: Input Tensor
      name : Optional name for the operation

   Returns:
      Tensor after ReLU operation
   """
   if name is None:
      name = "relu"

   with tf.variable_scope(name):
      return tf.nn.relu(input)

def tanh(input, name=None):
   """Tanh activation

   Args:
      input: Input Tensor
      name : Optional name for the operation

   Returns:
      Tensor after Tanh operation
   """
   if name is None:
      name = "tanh"

   with tf.variable_scope(name):
      return tf.nn.tanh(input)

def sigmoid(input, name=None):
   """Sigmoid activation

   Args:
      input: Input Tensor
      name : Optional name for the operation

   Returns:
      Tensor after Sigmoid operation
   """
   if name is None:
      name = "sigmoid"

   with tf.variable_scope(name):
      return tf.nn.sigmoid(input)

def lrelu(input, alpha=0.2, name=None):
   """Leaky ReLU

   Args:
      input: Input Tensor
      alpha: Slope for negative values
      name : Optional name for the operation

   Returns:
      Tensor after Learky ReLU operation
   """
   if name is None:
      name = "lrelu"

   with tf.variable_scope(name):
      return tf.maximum(input, alpha * input)

def batch_normalize(input, is_training, reuse=False, name=None):
   """Applies batch normalization

   Args:
      input      : Input Tensor
      is_training: Operation is in train / test time
      reuse      : Whether to reuse the batch statistics
      name       : Optional name for the operation

   Returns:
      Tensor after batch normalization operation
   """
   if name is None:
      name = "BatchNorm"

   with tf.variable_scope(name, reuse=reuse):
      output = tf.contrib.layers.batch_norm(
         inputs=input,
         center=True,
         scale=True,
         is_training=is_training)
      return output


def conv_bn_relu(input, kernel, out_channels, is_training, stride=1,
                 name=None, reuse=False,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 bias_constant=0.01):
   """Applies series of operations of `conv`->`batch_norm`->`relu`

   Args:
      input        : Input Tensor
      kernel       : filter's width (= filter's height)
      out_channels : Number of filters
      is_training  : Operation is in train / test time
      stride       : stride of the filter
      name         : Optional name for the operation
      reuse        : Whether to reuse this conv layer's weights and biases
      initializer  : Type of weight matrix initializer
      bias_constant: Constant value of bias

   Returns:
      Tensor after the series of operations
   """
   with tf.variable_scope(name+'_block', reuse=reuse):
      conv_ac = conv2d(input, kernel, out_channels, stride, name, reuse)
      conv_bn = batch_normalize(conv_ac, is_training, reuse=reuse)
      conv_rl = relu(conv_bn)
      return conv_rl


def conv_bn_lrelu(input, kernel, out_channels, is_training, stride=1,
                  name=None, reuse=False,
                  initializer=tf.contrib.layers.xavier_initializer(),
                  bias_constant=0.01):
   """Applies series of operations of `conv`->`batch_norm`->`leaky_relu`

   Args:
      input        : Input Tensor
      kernel       : filter's width (= filter's height)
      out_channels : Number of filters
      is_training  : Operation is in train / test time
      stride       : stride of the filter
      name         : Optional name for the operation
      reuse        : Whether to reuse this conv layer's weights and biases
      initializer  : Type of weight matrix initializer
      bias_constant: Constant value of bias

   Returns:
      Tensor after the series of operations
   """
   with tf.variable_scope(name+'_block', reuse=reuse):
      conv_ac = conv2d(input, kernel, out_channels, stride, name, reuse)
      conv_bn = batch_normalize(conv_ac, is_training, reuse=reuse)
      conv_rl = leaky_relu(conv_bn)
      return conv_rl


def dconv_bn_relu(input, kernel, out_channels, out_shape, is_training,
                  batch_size, stride=1, name=None, reuse=False,
                  initializer=tf.contrib.layers.xavier_initializer(),
                  bias_constant=0.01):
   """Applies series of operations of `conv`->`batch_norm`->`relu`

   Args:
      input        : Input Tensor
      kernel       : filter's width (= filter's height)
      out_channels : Number of filters
      out_shape    : Shape of the output tensor
      is_training  : Operation is in train / test time
      batch_size   : Batch Size
      stride       : stride of the filter
      name         : Optional name for the operation
      reuse        : Whether to reuse this conv layer's weights and biases
      initializer  : Type of weight matrix initializer
      bias_constant: Constant value of bias

   Returns:
      Tensor after the series of operations
   """
   with tf.variable_scope(name+'_block', reuse=reuse):
      conv_ac = deconv(input, kernel, out_shape, out_channels, stride,
                       name, reuse, batch_size=batch_size)
      conv_bn = batch_normalize(conv_ac, is_training, reuse=reuse)
      conv_rl = relu(conv_bn)
      return conv_rl


def dconv_bn_lrelu(input, kernel, out_channels, out_shape, is_training,
                   batch_size, stride=1, name=None, reuse=False, 
                   initializer=tf.contrib.layers.xavier_initializer(),
                   bias_constant=0.01):
   """Applies series of operations of `conv`->`batch_norm`->`leaky_relu`

   Args:
      input        : Input Tensor
      kernel       : filter's width (= filter's height)
      out_channels : Number of filters
      out_shape    : Shape of the output tensor
      is_training  : Operation is in train / test time
      batch_size   : Batch Size
      stride       : stride of the filter
      name         : Optional name for the operation
      reuse        : Whether to reuse this conv layer's weights and biases
      initializer  : Type of weight matrix initializer
      bias_constant: Constant value of bias

   Returns:
      Tensor after the series of operations
   """
   with tf.variable_scope(name+'_block', reuse=reuse):
      conv_ac = deconv(input, kernel, out_shape, out_channels, stride,
                       name, reuse, batch_size=batch_size)
      conv_bn = batch_normalize(conv_ac, is_training, reuse=reuse)
      conv_rl = leaky_relu(conv_bn)
      return conv_rl


def residual_block(input, output_channels, is_training, stride=1,
                   first_block=False, name=None, reuse=False):
   """Builds a residual block by applying the following operations:
      batch_norm -> relu -> conv

   Args:
      input        : Input Tensor
      out_channels : Number of filters
      is_training  : Operation is in train / test time
      stride       : stride of the filter
      first_block  : True if this block is the first block of the network
      name         : Optional name for the operation
      reuse        : Whether to reuse this conv layer's weights and biases

   Returns:
      A residual block
   """
   if name is None:
      name = "residual_block"

   input_channels = input.get_shape().as_list()[-2]
   with tf.variable_scope(name, reuse=reuse):
      # First conv block 
      with tf.variable_scope("block_1"):
         conv1_bn = batch_normalize(input, is_training, reuse=reuse)
         conv1_ac = relu(conv1_bn)
         conv1 = conv2d(conv1_ac, kernel=3, stride=stride, name="conv1",
                        reuse=reuse)
      # Second conv block
      with tf.variable_scope("block_2"):
         conv2_bn = batch_normalize(conv1, is_training, reuse=reuse)
         conv2_ac = relu(conv2_bn)
         conv2 = conv2d(conv2_ac, kernel=3, stride=stride, name="conv2",
                        reuse=reuse)
      if conv2.get_shape().as_list()[-1] != input.get_shape().as_list()[-1]:
         raise ValueError('Output and input channels do not match')
      else:
         output = input + conv2
      return output
   
def add_layers(layer_1, layer_2, name=None):
   """Adds two layers element wise

   Args:
      layer_1: The first layer
      layer_2: The second layer
      name   : Optional name for the operation

   Returns:
      Layer after addition of the given two
   """
   if name is None:
      name = 'add'
   l1_shape = layer_1.get_shape().as_list()[1:]
   l2_shape = layer_2.get_shape().as_list()[1:]
   assert(l1_shape == l2_shape), "Shapes {} and {} are not equal\
   ".format(l1_shape, l2_shape)
   with tf.variable_scope(name):
      return tf.add(layer_1, layer_2)

def activation_summary(tensor):
   """Write the summary of a tensor
   
   Args:
      tensor: Creates summary for this tensor
   """
   tensor_name = tensor.op.name
   tf.summary.histogram(tensor_name+'/activation', tensor)
   tf.summary.scalar(tensor_name+'/sparsity', tf.nn.zero_fraction(tensor))
