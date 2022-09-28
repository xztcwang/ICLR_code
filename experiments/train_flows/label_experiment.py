import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append("/home/tkw5356/GCFlow/")
import pandas as pd
from torch.utils.data import DataLoader
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier
from oil.model_trainers.piModel import PiModel
from oil.model_trainers.vat import Vat
from oil.datasetup.datasets import CIFAR10
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.img_classifiers.networkparts import layer13
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice, imap
from oil.tuning.study import Study, train_trial
from flow_ssl.data.nlp_datasets import AG_News,YAHOO
from flow_ssl.data import GAS, HEPMASS, MINIBOONE
from flow_ssl.data import CORA,CITESEER,TEXAS,PUBMED,CORA_SPLIT
from torchvision import transforms
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from collections import defaultdict
from oil.model_trainers.piModel import PiModel
from oil.model_trainers.classifier import Classifier
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam,AdamW
from oil.utils.utils import LoaderTo, cosLr, islice, dmap, FixedNumpySeed
from oil.tuning.study import train_trial
from oil.datasetup.datasets import CIFAR10, split_dataset
from oil.tuning.args import argupdated_config
from functools import partial
from train_semisup_text_baselines import SmallNN
from oil.tuning.args import argupdated_config
import copy
#import flow_ssl.data.nlp_datasets as nlp_datasets
import flow_ssl.data as tabular_datasets

#import train_semisup_flowgmm_tabular as flows
import train_semisup_flowgmm_graph as graphflows
import train_semisup_text_baselines as archs
import oil.model_trainers as trainers
from oil.utils.mytqdm import tqdm

seed_start=1
experiments_num=10
seeds = list(range(seed_start, seed_start+experiments_num))

# from flow_ssl.data.sample import Sampler
# sampler = Sampler("citeseer", '/home/tkw5356/flowgmm_config_gauss/data/graph_datasets', "semi")
# device = torch.device('cpu')
# labels, idx_train, idx_val, idx_remain, idx_test = sampler.get_label_and_idxes(device)

def makeTrainer(*, seed, gpu=2, dataset=CORA_SPLIT, ptscls=2,
                network=graphflows.NSFGraphWPrior,
                num_epochs=15,
                inner_epochs=3,
                bs=2708,#bs=2708, for cora, 3327 for citeseer, 19717 for pubmed
                lr=1e-3, optim=AdamW, trainer=Classifier,
                split=None,
                base_transform_type='affine-coupling',
                linear_transform_type='svd',
                net_config={'hidden_dim':1024,
                            'flow_layers':4,
                            'tail_bound':4,
                            'num_bins':8,
                            'num_transform_blocks':2,
                            'use_batch_norm':0,
                            'dropout_ratio':0.1,
                            'apply_unconditional_transform':1},
                gauss_config={'means_r':1.6, 'cov_std': 1.1},
                opt_config={'weight_decay':0.0005},
                trainer_config={'log_dir':os.path.expanduser('~/tb-experiments/UCI/'),
                                'log_args':{'minPeriod':.1, 'timeFrac':3/10},
                                'grad_norm_clip_value':50,
                                'unlab_weight':0.2},
                save=False):
    # Prep the datasets splits, model, and dataloaders
    if split is None:
        if dataset==CORA:
            split = {'train': 140, 'val': 500, 'remain': 1068, 'test': 1000}
        if dataset==CITESEER:
            split = {'train': 120, 'val': 500, 'remain': 1692, 'test': 1000}
        if dataset == PUBMED:
            split = {'train': 60, 'val': 500, 'remain': 18157, 'test': 1000}
        if dataset==TEXAS:
            split = {'train': 87, 'val': 58, 'test': 37}
        if dataset==CORA_SPLIT:
            train_num = ptscls*7
            remain_num = 140 + 1068 - train_num
            split = {'train': train_num, 'val': 500, 'remain': remain_num, 'test': 1000}


    with FixedNumpySeed(0):
        datasets = split_dataset(dataset(), splits=split)
        datasets['train'] = dataset(part='train', ptscls=ptscls)
        datasets['val'] = dataset(part='val', ptscls=ptscls)
        if dataset is not TEXAS:
            datasets['remain'] = dataset(part='remain', ptscls=ptscls)
        datasets['test'] = dataset(part='test', ptscls=ptscls)
        datasets['_unlab'] = dataset(part='_unlab', ptscls=ptscls)
        datasets['all'] = dataset(part='all', ptscls=ptscls)

        datasets['_unlab'] = dmap(lambda mb: mb[0], datasets['_unlab'])
        datasets['all'] = dmap(lambda mb: mb[0], datasets['all'])
        # datasets['test'] = dataset(train=False)
        # print(datasets['test'][0])
    # device = torch.device(device)
    #seed = 42
    #device = torch.device("cuda")
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    model = network(base_transform_type=base_transform_type,
                    linear_transform_type=linear_transform_type,
                    num_classes=datasets['train'].num_classes,
                    dim_in=datasets['train'].dim,
                    device=device,
                    trainloader=datasets['all'],
                    **net_config, **gauss_config).to(device)
    dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, len(datasets[k])), shuffle=(k == 'train'),
                                          num_workers=0, pin_memory=False), device) for k, v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)  # lambda e:1#
    return trainer(model, device, dataloaders, opt_constr, lr_sched, dataset, seed, **trainer_config)

if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    cfg = argupdated_config(defaults, namespace=(tabular_datasets, graphflows, archs, trainers))
    exp_id=0
    best_acc_list=[]
    best_macrof1_list=[]
    MI_list = []
    SScore_list = []


    for seed in tqdm(seeds):
        print("Experiment: {}".format(exp_id))
        trainer = makeTrainer(seed=seed, **cfg)
        trainer.dynamic_train_nsf(cfg['num_epochs'], cfg['inner_epochs'])
        best_acc_list.append(trainer.bestacc)
        best_macrof1_list.append(trainer.macro_f1)
        MI_list.append(trainer.MI)
        SScore_list.append(trainer.SScore)
        exp_id=exp_id+1
    best_acc_mean = np.mean(best_acc_list)
    best_acc_std = np.std(best_acc_list)
    best_macrof1_mean = np.mean(best_macrof1_list)
    best_macrof1_std = np.std(best_macrof1_list)

    MI_mean = np.mean(MI_list)
    MI_std = np.std(MI_list)
    SScore_mean = np.mean(SScore_list)
    SScore_std = np.std(SScore_list)

    #MI_mean = np.mean(MI_list)
    #MI_std = np.std(MI_list)
    print("=================\n")
    print("exp_list: \n")
    for i in range(experiments_num):
        print("exp:{}, acc: {:.3f}, silhouette: {:.3f}, mi: {:.3f}\n".format(i, best_acc_list[i],SScore_list[i],MI_list[i]))

    print("=================\n")
    print("Classification Accuracy: {:.3f} +- {:.3f}, Macro F1: {:.3f} +- {:.3f}\n".format(best_acc_mean,
                                                                                           best_acc_std,
                                                                                           best_macrof1_mean,
                                                                                           best_macrof1_std))
    print("=================\n")
    print("Clustering MI: {:.3f} +- {:.3f}, SScore: {:.3f} +- {:.3f}\n".format(MI_mean,
                                                                               MI_std,
                                                                               SScore_mean,
                                                                               SScore_std))
    print("=================")

    #print("Clustering Mutual Information: {:.3f} +- {:.3f}".format(MI_mean, MI_std))

