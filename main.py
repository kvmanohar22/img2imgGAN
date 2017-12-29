import pprint
import tensorflow as tf

from utils import *
from logger import log_config
from config import FLAGS

import utils
import nnet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(_):
   """Begins the execution of the program

   Args:
      _ : Tensorflow flags app instance
   """
   priliminary_checks(FLAGS)
   idx = get_runid(FLAGS)
   create_rundirs(FLAGS, idx)
   dump_model_params(FLAGS)

   log_config(idx, FLAGS.__flags)

   if FLAGS.archi:
      net = nnet.Model(FLAGS, False)
      net.test_graph()
      exit()

   if FLAGS.create != "":
      dataset = utils.Dataset(FLAGS)
      dataset.create_records(FLAGS.create)
      exit()

   FLAGS.h = 600 if FLAGS.dataset == 'maps' else 256
   FLAGS.w = FLAGS.h

   if FLAGS.train:
      print 'Summary of the model:'
      pp = pprint.PrettyPrinter()
      pp.pprint(FLAGS.__flags)
      net = nnet.Model(FLAGS, FLAGS.train)
      print 'Training the network...'
      net.train()
      print 'Done training the network...'
   else:
      print 'Testing the model...'
      net = nnet.Model(FLAGS, False)
      net.test(FLAGS.image)

if __name__ == '__main__':
   try:
      tf.app.run()
   except Exception as E:
      print E
