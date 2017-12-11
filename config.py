import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

# Data
tf.app.flags.DEFINE_string('root_dir', os.getcwd(), """Base Path""")
tf.app.flags.DEFINE_string('dataset_dir', os.path.join(FLAGS.root_dir, 'data'), """Path to data""")
tf.app.flags.DEFINE_integer('c', 1, """Number of input channels of images""")
tf.app.flags.DEFINE_string('dataset', "mnist", """mnist or CIFAR or imagenet""")
tf.app.flags.DEFINE_boolean('numpy_rec', False, """Are numpy records of data complete?""")

# Training
tf.app.flags.DEFINE_integer('batch_size', 64, """Batch size""")
tf.app.flags.DEFINE_integer('MAX_iterations', 1000, """Max iterations for training""")
tf.app.flags.DEFINE_integer('ckpt_frq', 100, """Frequency at which to checkpoint the model""")
tf.app.flags.DEFINE_integer('train_size', 10000, """The total training size""")
tf.app.flags.DEFINE_integer('display', 1, """Display log of progress""")
tf.app.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.app.flags.DEFINE_float('lr_decay', 0.9, """Learning rate decay factor""")
tf.app.flags.DEFINE_float('base_lr', 1e-6, """Base learning rate for VAE""")
tf.app.flags.DEFINE_boolean('train', False, """Training or testing""")
tf.app.flags.DEFINE_boolean('resume', False, """Resume the training ?""")

# Model Saving
tf.app.flags.DEFINE_string('sample_dir', os.path.join(FLAGS.root_dir,'logs', "sample"), """Generate sample images""")
tf.app.flags.DEFINE_string('summary_dir', os.path.join(FLAGS.root_dir, 'logs', "summary"), """Summaries directory including checkpoints""")
