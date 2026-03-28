import os
import warnings

import torch
import torchvision.transforms as transforms



class Config:
    mode = "CL"


    tr_list = 'dataset/dsn/train.lst'
    dev_list = 'dataset/dsn/dev.lst'
    checkpoint_root = 'asr_res_model/dns/dccrn'
    test_0_list = 'dataset/dsn/test_0.lst'

    batch_size = 16
    lr = 0.001  # learning_rate
    lr_decay = 0.1
    weight_decay = 1e-5
    verbose_inter = 500
    max_epoch = 40
    save_inter = 5
    device_ids = [0,0]
    device = device_ids[0]
    sr = 16000
    dim = 4 * 16000

    min_sisnr = 99999999

    #test
    best_path = 'asr_res_model/dns/dccrn/DCCRN_CL_25.pth'




