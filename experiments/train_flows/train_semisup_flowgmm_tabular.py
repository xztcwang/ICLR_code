import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from oil.model_trainers.classifier import Classifier,Trainer
from oil.utils.losses import softmax_mse_loss, softmax_mse_loss_both
from oil.utils.utils import Eval, izip, icycle,imap, export
#from .schedules import sigmoidConsRamp
import flow_ssl
import utils
from flow_ssl import FlowLoss
from flow_ssl.realnvp import RealNVPTabular
from flow_ssl.realnvp import RealNVPGraph
from flow_ssl.nsf import NSFGraph
from flow_ssl.distributions import SSLGaussMixture
from flow_ssl.tsne import plot_tsne
from scipy.spatial.distance import cdist

@export
#def RealNVPTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=.8,cov_std=1.,nperlayer=1,acc=0.9):
#def RealNVPTabularWPrior(num_classes,dim_in,device,coupling_layers,k,means_r=0.73,cov_std=2.97,nperlayer=1,acc=0.9, gauss_trainable=True, trainloader=None, **gcn_config):
def RealNVPTabularWPrior(num_classes, dim_in, device, coupling_layers, k, means_r=0.86, cov_std=2.76, nperlayer=1,
                         acc=0.9, gauss_trainable=True, trainloader=None, **gcn_config):
    #print(f'Instantiating means with dimension {dim_in}.')
    #device = torch.device('cuda')
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    model = RealNVPGraph(num_coupling_layers=coupling_layers, in_dim=dim_in, hidden_dim=k, n_class=num_classes, **gcn_config)
    #model = RealNVPTabular(num_coupling_layers=coupling_layers,in_dim=dim_in,hidden_dim=k,num_layers=1,dropout=True)#*np.sqrt(1000/dim_in)/3
    #dist_scaling = np.sqrt(-8*np.log(1-acc))#np.sqrt(4*np.log(20)/dim_in)#np.sqrt(1000/dim_in)
    if num_classes ==2:
        means = utils.get_means('random',r=means_r,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
        #means = torch.zeros(2,dim_in,device=device)
        #means[0,1] = 3.75
        dist = 2*(means[0]**2).sum().sqrt()
        means[0] *= 7.5/dist
        means[1] = -means[0]
        # means[0] /= means[0].norm()
        # means[0] *= dist_scaling/2
        # means[1] = - means[0]
        model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
        means_np = means.cpu().numpy()
    else:
        #means = utils.get_means('random',r=means_r*.7,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
        idx_train = trainloader.idx_train
        labels_train = trainloader.Y[idx_train]

        means = utils.get_means_graph('random',r=means_r*.7,num_means=num_classes, trainloader=trainloader,train_idx=idx_train,shape=(dim_in),device=device)
        # means = utils.get_means_graph('from_z', r=means_r * .7, num_means=num_classes, trainloader=trainloader,
        #                              train_idx=idx_train, shape=(dim_in), device=device, net=model)

        # if gauss_trainable:
        #     means_np = means.cpu().numpy()
        #     inv_cov_std_np = inv_cov_std.cpu().numpy()
        #     means = torch.tensor(means_np, requires_grad=gauss_trainable)
        #     inv_cov_std = torch.tensor(inv_cov_std_np, requires_grad=gauss_trainable)

        model.prior = SSLGaussMixture(trainloader, means, means_r*.7, inv_cov_std,device=device)
        means_np = means.detach().cpu().numpy()
    print("Pairwise dists:", cdist(means_np, means_np))
    # means_np = model.prior.means.detach().cpu().numpy()
    # inv_cov_stds_np = model.prior.inv_cov_stds.detach().cpu().numpy()
    # means = torch.tensor(means_np, requires_grad=True)
    # inv_cov_stds = torch.tensor(inv_cov_stds_np, requires_grad=True)
    # model.prior.means=means.to(device).detach()
    # model.prior.inv_cov_stds=inv_cov_stds.to(device).detach()
    #model.prior.means=model.prior.means.to(device)
    #model.prior.inv_cov_stds=model.prior.inv_cov_stds.to(device)
    return model



# @export
# def NSFTabularWPrior(num_classes, dim_in, device, coupling_layers, hdim=128, KK=5, BB=3, means_r=0.86, cov_std=2.76, nperlayer=1,
#                      acc=0.9, trainloader=None, flowchoice=None, **gcn_config):
#     inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
#     flow = eval(flowchoice)
#     flows = [flow(dim=dim_in, K = KK, B = BB, hidden_dim=hdim) for _ in range(coupling_layers)]
#     idx_train = trainloader.idx_train
#     means = utils.get_means_graph('random', r=means_r * .7, num_means=num_classes, trainloader=trainloader,
#                                   train_idx=idx_train, shape=(dim_in), device=device)
#     prior = SSLGaussMixture(trainloader, means, means_r * .7, inv_cov_std, device=device)
#     model = NSFGraph(prior, flows)
#     means_np = means.detach().cpu().numpy()
#     print("Pairwise dists:", cdist(means_np, means_np))
#     return model








def ResidualTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=1.,cov_std=1.,nperlayer=1,acc=0.9):
    #print(f'Instantiating means with dimension {dim_in}.')
    device = torch.device('cuda')
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    model = TabularResidualFlow(in_dim=dim_in,hidden_dim=k,num_per_block=coupling_layers)#*np.sqrt(1000/dim_in)/3
    dist_scaling = np.sqrt(-8*np.log(1-acc))
    means = utils.get_means('random',r=means_r*dist_scaling,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
    means[0] /= means[0].norm()
    means[0] *= dist_scaling/2
    means[1] = - means[0]
    model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
    means_np = means.cpu().numpy()
    #print("Pairwise dists:", cdist(means_np, means_np))
    return model

@export
class SemiFlow(Trainer):
    def __init__(self, *args, unlab_weight=1.,cons_weight=3.,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'unlab_weight':unlab_weight,'cons_weight':cons_weight})
        #self.dataloaders['train'] = izip(icycle(self.dataloaders['train']),self.dataloaders['_unlab'])


        self.idx_train = self.dataloaders['all'].dataset.idx_train.to(self.device)
        self.idx_val = self.dataloaders['all'].dataset.idx_val.to(self.device)
        self.idx_test = self.dataloaders['all'].dataset.idx_test.to(self.device)
        self.idx_remain=self.dataloaders['all'].dataset.idx_remain.to(self.device)
        self.idx_all = self.dataloaders['all'].dataset.indeces_use.to(self.device)
        self.adj_mat = self.dataloaders['all'].dataset.train_adj.to(self.device)
        self.idx_unlab = torch.cat([self.idx_val,self.idx_remain,self.idx_test], dim=0)
        self.dataloaders['train'] = izip(icycle(self.dataloaders['train']), self.dataloaders['all'])
        self.train_labels=self.dataloaders['all'].dataset.Y[self.idx_train].to(self.device)
        self.val_labels=self.dataloaders['all'].dataset.Y[self.idx_val].to(self.device)
        self.test_labels=self.dataloaders['all'].dataset.Y[self.idx_test].to(self.device)
        self.all_labels=self.dataloaders['all'].dataset.Y.to(self.device)



    # def loss(self, minibatch):
    #     (x_lab, y_lab), x_unlab = minibatch
    #     a = float(self.hypers['unlab_weight'])
    #     b = float(self.hypers['cons_weight'])
    #     flow_loss = self.model.nll(x_lab,y_lab).mean() + a*self.model.nll(x_unlab).mean()
    #     # with torch.no_grad():
    #     #     unlab_label = self.model.prior.classify(self.model(x_unlab)).detach()
    #     # cons_loss = self.model.nll(x_unlab,unlab_label).mean()
    #     return flow_loss#+b*cons_loss
    def loss(self, minibatch):
        #(x_lab, y_lab), x_unlab = minibatch
        x_adj_mat=(minibatch[1], self.adj_mat)
        a = float(self.hypers['unlab_weight'])
        b = float(self.hypers['cons_weight'])
        #c = float(self.hypers['det_weight'])
        label_term, _,_,_ = self.model.nll(x_adj_mat, self.idx_train, self.train_labels)
        label_term = label_term.mean()
        unlabel_term, _,_,_ = self.model.nll(x_adj_mat, self.idx_unlab, y=None)
        unlabel_term = unlabel_term.mean()
        _, logdet, _, matdet_list = self.model.nll(x_adj_mat, self.idx_all)
        logdet = (logdet / len(self.idx_all)).cpu().data.numpy()
        #logdet = logdet / len(self.idx_all)
        #flow_loss = label_term+a*unlabel_term+a*logdet
        flow_num = len(matdet_list)
        logdet_train = 0
        logdet_unlab = 0
        for flow_id in range(flow_num):
            logdet_train -= matdet_list[flow_id][self.idx_train, :].sum()/ len(self.idx_all)
            logdet_unlab -= matdet_list[flow_id][self.idx_unlab, :].sum()/ len(self.idx_all)
        flow_loss = label_term+logdet_train + a * (unlabel_term+logdet_unlab)



        # flow_loss = self.model.nll(x_adj_mat, self.idx_train, self.train_labels).mean() + \
        #             a*self.model.nll(x_adj_mat, self.idx_unlab, y=None).mean()
        # with torch.no_grad():
        #     unlab_label = self.model.prior.classify(self.model(x_unlab)).detach()
        # cons_loss = self.model.nll(x_unlab,unlab_label).mean()
        return flow_loss#+b*cons_loss

    def step(self, minibatch):
        self.optimizer.zero_grad()
        loss = self.loss(minibatch)
        loss.backward()
        utils.clip_grad_norm(self.optimizer, 100)
        self.optimizer.step()
        return loss
    # def step(self, minibatch, adj_mat):
    #     self.optimizer.zero_grad()
    #     loss = self.loss(minibatch, adj_mat)
    #     loss.backward()
    #     utils.clip_grad_norm(self.optimizer, 100)
    #     self.optimizer.step()
    #     return loss
        
    # def logStuff(self, step, minibatch=None, adj_mat=None):
    #     bpd_func = lambda mb: (self.model.nll(mb).mean().cpu().data.numpy()/mb.shape[-1] + np.log(256))/np.log(2)
    #     acc_func = lambda mb: self.model.prior.classify(self.model(mb[0])).type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
    #     metrics = {}
    #     with Eval(self.model), torch.no_grad():
    #         #metrics['Train_bpd'] = self.evalAverageMetrics(self.dataloaders['unlab'],bpd_func)
    #         metrics['val_bpd'] = self.evalAverageMetrics(imap(lambda z: z[0],self.dataloaders['val']),bpd_func)
    #         metrics['Train_Acc'] = self.evalAverageMetrics(self.dataloaders['Train'],acc_func)
    #         metrics['val_Acc'] = self.evalAverageMetrics(self.dataloaders['val'],acc_func)
    #         metrics['test_Acc'] = self.evalAverageMetrics(self.dataloaders['test'],acc_func)
    #         if minibatch:
    #             metrics['Unlab_loss(mb)']=self.model.nll(minibatch[1]).mean().cpu().data.numpy()
    #     self.logger.add_scalars('metrics',metrics,step)
    #     super().logStuff(step, minibatch)
    def logStuff(self, step, minibatch=None):
        #bpd_func = lambda mb: (self.model.nll(mb).mean().cpu().data.numpy()/mb.shape[-1] + np.log(256))/np.log(2)
        bpd_func = lambda mb,adj, idx: (self.model.nll(mb, adj, idx).mean().cpu().data.numpy() / mb.shape[-1] + np.log(256)) / np.log(2)
        #acc_func = lambda mb,adj,idx: self.model.prior.classify((self.model(mb[0],adj))[idx]).type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
        train_acc_fun = lambda mb, mb1_adj, idx: self.model.prior.classify((self.model(mb1_adj))[0][idx]).type_as(mb[0][1]).eq(
            mb[0][1]).cpu().data.numpy().mean()
        val_acc_fun = lambda mb_adj,idx,label:self.model.prior.classify((self.model(mb_adj))[0][idx]).type_as(label).eq(
            label).cpu().data.numpy().mean()
        metrics = {}
        with Eval(self.model), torch.no_grad():
            #metrics['Train_bpd'] = self.evalAverageMetrics(self.dataloaders['unlab'],bpd_func)

            #metrics['val_bpd'] = self.evalAverageMetrics(imap(lambda z: z[0],self.dataloaders['val']),bpd_func)
            #metrics['Train_Acc'] = self.evalAverageMetrics(self.dataloaders['Train'],acc_func)
            #metrics['Train_Acc'] = self.evalAverageMetrics(self.dataloaders['train'],self.adj_mat, self.idx_train, train_acc_fun)
            metrics['Train_Acc'] = self.evalAverageMetrics(self.dataloaders['all'],self.adj_mat,self.idx_train,val_acc_fun,self.train_labels)
            metrics['val_Acc'] = self.evalAverageMetrics(self.dataloaders['all'], self.adj_mat, self.idx_val, val_acc_fun, self.val_labels)
            metrics['test_Acc'] = self.evalAverageMetrics(self.dataloaders['all'], self.adj_mat, self.idx_test,
                                                         val_acc_fun, self.test_labels)
            #metrics['val_Acc'] = self.evalAverageMetrics(self.dataloaders['val'],self.adj_mat, self.idx_val, acc_func)
            #metrics['test_Acc'] = self.evalAverageMetrics(self.dataloaders['test'],acc_func)
            if minibatch:
                mb1_adj=(minibatch[1], self.adj_mat)
                prior_ll_train_, _,_, _ = self.model.nll(mb1_adj,self.idx_train, minibatch[0][1])
                if torch.isinf(prior_ll_train_.mean()):
                    print("Inf here")
                prior_ll_train = prior_ll_train_.mean().cpu().data.numpy()
                prior_ll_unlab_, _,_,_ = self.model.nll(mb1_adj, self.idx_unlab)
                if torch.isinf(prior_ll_unlab_.mean()):
                    print("Inf here")
                prior_ll_unlab = prior_ll_unlab_.mean().cpu().data.numpy()
                _, logdet, z, matdet_list = self.model.nll(mb1_adj, self.idx_all)
                logdet = (logdet/len(self.idx_all)).cpu().data.numpy()
                #metrics['labeled_loss'] = prior_ll_train
                #metrics['unlabeled_loss'] = prior_ll_unlab
                #metrics['logdet']=logdet

                #metrics['Train_loss(mb)'] = self.model.nll(mb1_adj,self.idx_train, minibatch[0][1]).mean().cpu().data.numpy()

                #if self.epoch==self.totalep:
                #    plot_tsne(z0.cpu().detach().numpy(),self.all_labels.cpu().numpy(),"embedding")

                #metrics['Unlab_loss(mb)']=self.model.nll(mb1_adj, self.idx_all).mean().cpu().data.numpy()
            # plotting t-sne
            #     self.plot_sne_()
        self.logger.add_scalars('metrics', metrics, step+self.inner_epoch/10)
        super().logStuff(step+self.inner_epoch/10, minibatch)
        if step == self.totalep - 1:
            self.finalacc= metrics['test_Acc']
        # if (step+1)%50 ==0 and step>0:
        #     self.plot_sne_(step)
        # if step%50==0 and step>0:
        #     self.plot_sne_(step)



    def plot_sne_(self,step):
        for _, mb in enumerate(self.dataloaders['all']):
            mb_adj = (mb, self.adj_mat)
            _, _, z0,_ = self.model.nll(mb_adj, self.idx_all)
            plot_tsne(z0.cpu().detach().numpy(), step, self.all_labels.cpu().numpy(), "embedding")

    #def get_z(self, minibatch):


from oil.tuning.study import Study, train_trial
import collections
import os
import utils
import copy
#from train_semisup_text_baselines import makeTabularTrainer
#from flowgmm_tabular_new import tabularTrial
from flow_ssl.data.nlp_datasets import AG_News
from flow_ssl.data import GAS, HEPMASS, MINIBOONE

# if __name__=="__main__":
#     trial(uci_hepmass_flowgmm_cfg)
    # thestudy = Study(trial,uci_hepmass_flowgmm_cfg,study_name='uci_flowgmm_hypers222_m__m_m')
    # thestudy.run(1,ordered=False)
    # covars = thestudy.covariates()
    # covars['test_Acc'] = thestudy.outcomes['test_Acc'].values
    # covars['dev_Acc'] = thestudy.outcomes['dev_Acc'].values
    #print(covars.drop(['log_suffix','saved_at'],axis=1))
    # print(thestudy.covariates())
    # print(thestudy.outcomes)

