import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
sample_dir = os.path.join(cwd,'logs', "sample")
summary_dir = os.path.join(cwd,'logs', "summary")

tf.app.flags.DEFINE_boolean('archi', False, """Test the architecture of the network?""")

# Data
tf.app.flags.DEFINE_string('root_dir', cwd, """Base Path (Default is the present working directory)""")
tf.app.flags.DEFINE_string('dataset_dir', data_dir, """Path to data""")
tf.app.flags.DEFINE_integer('h', 256, """Height of images""")
tf.app.flags.DEFINE_integer('w', 256, """Width of images""")
tf.app.flags.DEFINE_integer('c', 1, """Number of input channels of images""")
tf.app.flags.DEFINE_string('dataset', "maps", """edges2handbags/edges2shoes/facades/maps""")
tf.app.flags.DEFINE_string('create', "", """Create numpy records of the given dataset""")

# Training
tf.app.flags.DEFINE_integer('batch_size', 1, """Batch size""")
tf.app.flags.DEFINE_integer('MAX_iterations', 1000, """Max iterations for training""")
tf.app.flags.DEFINE_integer('ckpt_frq', 100, """Frequency at which to checkpoint the model""")
tf.app.flags.DEFINE_integer('train_size', 10000, """The total training size""")
tf.app.flags.DEFINE_integer('display', 1, """Display log of progress""")
tf.app.flags.DEFINE_integer('code_len', 8, """Length of latent dimension""")
tf.app.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.app.flags.DEFINE_float('lr_decay', 0.9, """Learning rate decay factor""")
tf.app.flags.DEFINE_float('base_lr', 0.0002, """Base learning rate for VAE""")
tf.app.flags.DEFINE_boolean('train', False, """Training or testing""")
tf.app.flags.DEFINE_boolean('resume', False, """Resume the training ?""")

# Loss specific
tf.app.flags.DEFINE_float('lambda_img', 10, """Parameter to balance the loss""")
tf.app.flags.DEFINE_float('lambda_latent', 0.5, """Parameter to balance the loss""")
tf.app.flags.DEFINE_float('lambda_kl', 0.01, """Parameter to balance the loss""")

# Model Saving
tf.app.flags.DEFINE_string('sample_dir', sample_dir, """Generate sample images""")
tf.app.flags.DEFINE_string('summary_dir', summary_dir, """Summaries directory including checkpoints""")

# Encoder
tf.app.flags.DEFINE_string('e_type', 'normal', """Type of the network, {normal or residual}""")
tf.app.flags.DEFINE_integer('e_layers', 3, """Number of layers in the encoder network""")
tf.app.flags.DEFINE_integer('e_kernels', 64, """Number of kernels for the first layer of encoder""")
tf.app.flags.DEFINE_string('e_nonlin', 'relu', """Type of non-linearity for the encoder network {relu or lrelu}""")
# Generator
tf.app.flags.DEFINE_string('where_add', 'input', """Where to concatenate the noise the generator network {input or all}""")
tf.app.flags.DEFINE_integer('g_layers', 3, """Number of layers in the generator network""")
tf.app.flags.DEFINE_integer('g_kernels', 64, """Number of kernels for the first layer of generator""")
tf.app.flags.DEFINE_string('g_nonlin', 'lrelu', """Type of non-linearity for the generator network {relu or lrelu}""")
# Discriminator
tf.app.flags.DEFINE_string('d_nonlin', 'lrelu', """Type of non-linearity for the discriminator network""")
tf.app.flags.DEFINE_boolean('d_usemulti', False, """Use multiple discriminators for Discriminator?""")

# Testing
tf.app.flags.DEFINE_string('ckpt', '', """Checkpoint to load to test the model""")
tf.app.flags.DEFINE_string('image', '', """Path of the input image to test the network""")
