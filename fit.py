import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import torch.nn.functional as F
from dataset import get_Dataloader
from log import _logger




class EarlyStopping:
    def __init__(self,logger, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger


    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.logger.debug(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, '_checkpoint.pth'))
        self.val_loss_min = val_loss


class fit(object):
    def __init__(self,args,net):
        # self.mode = MLPMixer(args.image_size,args.patch_size,args.hidden_dim,args.depth,dropout=args.dropout,num_classes=args.num_classes)
        self.mode = net
        # self.mode = kaggle_cnn()
        # self.mode = MLP()
        # self.mode = AlexNet()
        # self.mode = convnextv2()
        self.opt = torch.optim.Adam(self.mode.parameters(),lr=args.learning_rate)
        self.lossfunc = nn.CrossEntropyLoss()
        self.args = args
        local_time = time.strftime("%Y年%m月%d日 %H时%M分%S秒")
        if os.path.isdir('args.save_dir'+ f'{local_time}') is True:
            self.log = _logger(args.save_dir + f'{local_time}/log')
        else:
            os.mkdir(args.save_dir + f'{local_time}')
            self.log = _logger(args.save_dir + f'{local_time}/log')
        self.save_dir = args.save_dir + f'{local_time}/'
        if torch.cuda.is_available():
            self.mode.cuda()
        
        self.train_dataloader = get_Dataloader(mode='train',batch_size=args.batch_size)
        self.val_dataloader = get_Dataloader(mode='val',batch_size=args.batch_size)
        self.test_dataloader = get_Dataloader(mode='test',batch_size=args.batch_size)
        self.earlystopping = EarlyStopping(logger=self.log)


    def vis(self,conf,train_iter_loss,val_acc,val_loss):
        print(val_acc)
        _,ax = plt.subplots(2,2,figsize=(20,15))
        sns.heatmap(conf, annot=True, cmap='Blues',ax=ax[0][0],fmt='d')
        
        sns.lineplot(data=train_iter_loss,ax=ax[0][1])
        ax[0][1].set_xlabel('迭代次数')
        ax[0][1].set_ylabel('train_loss')
        ax[0][1].set_xlim(0,train_iter_loss.shape[0])

        sns.lineplot(data=val_acc,ax=ax[1][0])
        ax[1][0].set_xlabel('Epoch')
        ax[1][0].set_ylabel('准确率')
        ax[1][0].set_xlim(0,val_acc.shape[0])
        ax[1][0].set_ylim(0,1)

        sns.lineplot(data=val_loss,ax=ax[1][1])
        ax[1][1].set_xlabel('Epoch')
        ax[1][1].set_ylabel('val_loss')
        ax[1][1].set_xlim(0,val_loss.shape[0])
        plt.savefig(f'{self.save_dir}'+ 'vis.pdf',dpi=1200)

    def val(self):
        val_loss = []
        val_acc = []
        self.mode.eval()
        for data, label in self.val_dataloader:
            data = data.to(self.args.device)
            label = label.to(self.args.device)
            output = self.mode(data)
            loss = self.lossfunc(output,label)
            val_loss.append(loss.item())
            acc = torch.sum(torch.argmax(output,dim=-1) == torch.argmax(label,dim=-1))/output.shape[0]
            
        return np.average(val_loss) , acc.item() 

    def train(self):
        train_iter_loss = []
        val_acc_list = []
        val_loss_list = []
        iter = 0

        self.log.debug('='*10 +'Train' + '='*10)
        for i in range(self.args.epoch):
            train_loss = 0
            self.mode.train()
            for data , label in self.train_dataloader:
                iter += 1
                self.opt.zero_grad() 
                data = data.to(self.args.device) # 128,1,28,28
                label = label.to(self.args.device)
                output = self.mode(data)
                iter_loss = self.lossfunc(output,label)  
                train_loss += self.lossfunc(output,label).item() * data.shape[0]
                train_iter_loss.append(iter_loss.item())
                iter_loss.backward()
                self.opt.step()
            
            val_loss,val_acc = self.val()
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            self.log.debug(f'eopch:{i + 1} train_loss: {train_loss/len(self.train_dataloader.sampler):.2f}  val_loss:{val_loss}  val_acc:{val_acc}')

            self.earlystopping(val_acc, self.mode, self.save_dir)
            if self.earlystopping.early_stop:
                print("Early stopping")
                np.save(f'{self.save_dir}' + 'train_iter_loss',np.array(train_iter_loss))
                np.save(f'{self.save_dir}' + 'val_loss',np.array(val_loss_list))
                np.save(f'{self.save_dir}' + 'val_acc',np.array(val_acc_list))
                break

        np.save(f'{self.save_dir}' + 'train_iter_loss',np.array(train_iter_loss))
        np.save(f'{self.save_dir}' + 'val_loss',np.array(val_loss_list))
        np.save(f'{self.save_dir}' + 'val_acc',np.array(val_acc_list))
        

    def test(self):
        self.mode.load_state_dict(
            torch.load(
                os.path.join(self.save_dir + '_checkpoint.pth')))
        self.mode.eval()
        pred_label = []
        true_label = []
        acc_num = 0
        all_num = 0
        for data ,label in self.test_dataloader:
            data = data.to(self.args.device)
            output = self.mode(data)
            acc_num += torch.sum(torch.argmax(output.cpu().detach(),axis=-1) == torch.argmax(label,dim=-1))
            all_num += data.shape[0]
            pred_label.extend(list(np.array(np.argmax(output.cpu().detach(),axis=-1)).reshape(-1)))
            true_label.extend(list(np.array(torch.argmax(label,dim=-1)).reshape(-1)))
        self.log.debug(f'test_acc:{acc_num / all_num}')

        conf = metrics.confusion_matrix(pred_label,true_label)
        train_iter_loss = np.load(f'{self.save_dir}' + 'train_iter_loss.npy')
        val_loss = np.load(f'{self.save_dir}' + 'val_loss.npy')
        val_acc = np.load(f'{self.save_dir}' + 'val_acc.npy')
        self.vis(conf,train_iter_loss,val_acc,val_loss)
