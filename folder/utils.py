import numpy as np
import os
import os.path as osp
import argparse

Config ={}
Config['root_path'] = 'D:\\EE599 deep leaning\\project\\ifood-2019-fgvc6'
#Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''
Config['train_set'] = 'train_set'
#Config['test_set'] = 'valaditon_set.zip'
Config['valaditon_set'] = 'val_set'

Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 20
Config['batch_size'] = 64

Config['learning_rate'] = 0.001
Config['num_workers'] = 5