import json
import os

def log_config(idx, flags):
   """Writes the initial hyperparameters to a file
   """
   with open(os.path.join('logs/config_{}'.format(idx)), 'w') as fp:
      json.dump(flags, fp, indent=4)
