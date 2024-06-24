import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from models.xception import TransferModel
from Generator import synthesizer
from models.ConvNext import convnext_tiny,convnext_small,convnext_base,convnext_large,convnext_xlarge
from models.am_softmax import AMSoftmaxLoss
# from models.CPA import cpa
import cv2
from torchvision import transforms as T
import argparse
from PIL import Image
model_name = 'ProFake'
def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())



def quality_order_loss(pristine_quality, degraded_quality, margin=0.1):
    """
    Compute the quality order loss using a ranking loss, ensuring that the predicted quality score
    of a pristine sample is higher than that of a degraded sample by at least a margin.
    
    Args:
    - pristine_quality (Tensor): A tensor containing the probabilities of the positive class for the pristine samples.
    - degraded_quality (Tensor): A tensor containing the probabilities of the positive class for the degraded samples.
    - margin (float): The margin by which pristine_quality's score should exceed degraded_quality's score.
    
    Returns:
    - loss (Tensor): The computed quality order loss using ranking loss.
    """
    # Create a tensor of -1s to indicate that pristine_quality should be higher than degraded_quality
    target = torch.ones_like(pristine_quality[:, 1])  # Target tensor of 1s

    # Compute the ranking loss
    loss_fn = nn.MarginRankingLoss(margin=margin)
    loss = loss_fn(pristine_quality[:, 1], degraded_quality[:, 1], target)

    return loss

class Detector(nn.Module):
    def __init__(self):
        super(Detector,self).__init__()
        self.detector = convnext_base(2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # classification head
        self.fc_cls = nn.Linear(1024,2)
        

        # quality head
        self.q_head = nn.Sequential(nn.Linear(1024,2))


        
    def forward(self,im):
        fea, _ = self.detector(im,True)
        latent = fea[-1]
        latent = self.pool(latent).reshape(latent.shape[0],-1)
        
        score = self.fc_cls(latent)
        q = self.q_head(latent)
        
        return score, q, latent



class SampleNet(nn.Module):
    def __init__(self, num_bins=1000, embedding_dim=64, num_channels=3, image_size=(299, 299)):
        super(SampleNet, self).__init__()
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding = nn.Embedding(num_embeddings=num_bins, embedding_dim=embedding_dim)
        dropout = 0.5
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024 + embedding_dim * 2, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128,1)
            nn.Sigmoid()
        )
        self.fc[1].bias.data.fill_(-np.log(1 / 0.99 - 1))  # Inverse of sigmoid for 0.99
        
    def get_embed(self, values):
        indices = (values * (self.num_bins - 1)).long()
        return self.embedding(indices)

    def forward(self, dy, q, fea):
        embed_y = self.get_embed(dy)
        embed_q = self.get_embed(q)
        combined_features = torch.cat((fea, embed_y, embed_q), dim=1)
        w = self.fc(combined_features)
        
        # Pass the combined features through the fully connected layer
        w = torch.sigmoid(w)
        return w



class Model():
    def __init__(self, opt, logdir=None, train=True):
       
        if opt is not None:
            self.meta = opt.meta
            self.opt = opt
            self.ngpu = opt.ngpu

        self.writer = None
        self.logdir = logdir
        dropout = 0.5
        self.M = 1
        self.log_degradation = True
        self.model = DF()
        self.synthesizer = synthesizer()
        self.adapter = SampleNet()
        
    
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_criterion = AMSoftmaxLoss(gamma=0., m=0.45, s=30, t=1.)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.train = train
        self.l1loss = nn.MSELoss() # L2 norm
        

        params = get_params_groups(self.model,weight_decay = opt.wd)
        params_synthesizer = ([p for p in self.synthesizer.parameters()])
        params_cpa = ([p for p in self.adapter.parameters()])
        params_qc = ([p for p in self.qc.parameters()])
        
        if train:
            self.optimizer = optim.AdamW(params,lr=opt.lr,weight_decay=opt.wd)
            self.optimizer_synthesizer = optim.Adam(params_synthesizer, lr=opt.lr/4, betas=(opt.beta1, 0.999),
                                        weight_decay=opt.weight_decay)
            self.optimizer_adapter = optim.Adam(params_cpa, lr=opt.lr/4, betas=(opt.beta1, 0.999),
                                        weight_decay=opt.weight_decay)
        
    def define_summary_writer(self):
        if self.logdir is not None:
            # tensor board writer
            timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            log = '{}/{}/{}'.format(self.logdir, model_name, self.meta)
            log = log + '_{}'.format(timenow)
            print('TensorBoard log dir: {}'.format(log))

            self.writer = SummaryWriter(log_dir=log)


    def setTrain(self):
        self.model.train()
        self.synthesizer.train()
        self.adapter.train()
        self.qc.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path=None, synthesizer_path = 0, adapter_path = 0, qc_path = 0):
        if model_path !=0 and os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                for name in saved:
                    print(name)
                self.model.load_state_dict(saved)
            print('Model found in {}'.format(model_path))

        if synthesizer_path != 0 and os.path.isfile(synthesizer_path):
            saved = torch.load(synthesizer_path, map_location='cpu')
            suffix = synthesizer_path.split('.')[-1]
            if suffix == 'p':
                self.synthesizer.load_state_dict(saved.state_dict())
            else:
                self.synthesizer.load_state_dict(saved)


        if adapter_path != 0 and os.path.isfile(adapter_path):
            saved = torch.load(adapter_path, map_location='cpu')
            suffix = adapter_path.split('.')[-1]
            if suffix == 'p':
                self.adapter.load_state_dict(saved.state_dict())
            else:
                self.adapter.load_state_dict(saved)



    

        
    def save_ckpt(self, dataset, epoch, iters, save_dir, best=False):
        print("saving the model")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        mid_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)

        sub_dir = os.path.join(mid_dir, self.meta)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        subsub_dir = os.path.join(sub_dir, dataset)
        if not os.path.exists(subsub_dir):
            os.mkdir(subsub_dir)
        
        if best:
            ckpt_name = "epoch_{}_iter_{}_best.pth".format(epoch, iters)
            ckpt_name2 = "epoch_{}_iter_{}_best_syn.pth".format(epoch, iters)
            ckpt_name3 = "epoch_{}_iter_{}_best_ada.pth".format(epoch, iters)
        else:
            ckpt_name = "epoch_{}_iter_{}.pth".format(epoch, iters)
            ckpt_name2 = "epoch_{}_iter_{}_syn.pth".format(epoch, iters)
            ckpt_name3 = "epoch_{}_iter_{}_ada.pth".format(epoch, iters)
            
        save_path = os.path.join(subsub_dir, ckpt_name)
        save_path_ctrl = os.path.join(subsub_dir, ckpt_name2)
        save_path_ada = os.path.join(subsub_dir, ckpt_name3)
        
        print(save_path)
        if self.ngpu > 1:
            torch.save(self.model.module.state_dict(), save_path)
            torch.save(self.synthesizer.module.state_dict(), save_path_ctrl)
            torch.save(self.adapter.module.state_dict(), save_path_ada)
            
        else:
            torch.save(self.model.state_dict(), save_path)
            torch.save(self.synthesizer.state_dict(), save_path_ctrl)
            torch.save(self.adapter.state_dict(), save_path_ada)
        print("Checkpoint saved to {}".format(save_path))

    def optimize(self, img, label):
        log_prob = None
        device = torch.device("cuda")
        img = img.to(device)
        # Do the synthesize
        im_lq, k, ns, sdown = self.synthesizer(img)
        
        im_lq = im_lq.to(device)
        
        k = k.to(device)
        ns = ns.to(device)
        sdown = sdown.to(device)
        label = label.to(device)
        # print("The degree of degradation:")
        # print(k,ns,sdown)
        


        # Prediction
        
        out_h, q_hat_h, fea_h = self.model(img)
        out_l, q_hat_l, fea_l = self.model(im_lq)
        # print("The prediction:")
        # print(out_h, out_l)

        # print("The quality estimation:")
        # print(q_hat_h, q_hat_l)
        q_hat_h = q_hat_h.to(device)
        q_hat_l = q_hat_l.to(device)
        out_h = out_h.to(device)
        out_l = out_l.to(device)
        fea_h = fea_h.to(device)
        fea_l = fea_l.to(device)
        

        
        
        dy_h = torch.abs(label - out_h[:,1]).to(device)
        dy_l = torch.abs(label - out_l[:,1]).to(device)
        

        w_h = self.adapter(dy_h,q_hat_h[:,1],fea_h)
        w_l = self.adapter(dy_l,q_hat_l[:,1],fea_l)
        print("assign sample weights")
        print(w_h,w_l)

        


        # weighted cls loss
        loss_cls = self.cls_criterion(out_h,label,w_h) + self.cls_criterion(out_l,label,w_l)
        
        
        # quality order loss
        loss_order = quality_order_loss(q_hat_h, q_hat_l).mean()
        
        
        # adapter loss
        loss_adapter = (1 - w_h)**2 * dy_h + w_h**2 * max(self.M - dy_h, 0) + (1 - w_l)**2 * dy_l + w_l**2 * max(self.M - dy_l, 0)
        
        
        
        
        
        loss = loss_cls + loss_order + loss_adapter
        # print(loss, loss_cls, loss_order, loss_adapter)

        
        
        if self.train:
            self.optimizer_synthesizer.zero_grad()
            self.optimizer.zero_grad()
            self.optimizer_adapter.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_synthesizer.step()
            self.optimizer_adapter.step()
        
        
        return out_h, out_l, loss, loss_cls, loss_order, loss_adapter

    def inference(self, img, label):
        with torch.no_grad():
            score, _, _ = self.model(img)
            # loss_cls = self.cls_criterion(score, label).mean()
            return score
    



    def update_tensorboard(self, losses, step, acc_h=None,acc_l = None, datas=None, name='train'):
        assert self.writer
        if losses is not None:
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(f'{name}/Loss/{loss_name}', loss_value, global_step=step)
        # if loss is not None:
        #     loss_dic = {'Cls': loss}
        #     self.writer.add_scalars('{}/Loss'.format(name), tag_scalar_dict=loss_dic,
        #                             global_step=step)

        if acc_h is not None:
            self.writer.add_scalar('{}/Acc(Raw)'.format(name), acc_h, global_step=step)

        if acc_l is not None:
            self.writer.add_scalar('{}/Acc(Degraded)'.format(name), acc_l, global_step=step)
        
        if datas is not None:
            self.writer.add_pr_curve(name, labels=datas[:, 1].long(),
                                     predictions=datas[:, 0], global_step=step)

    def update_tensorboard_test_accs(self, accs, step, feas=None, label=None, name='test'):
        assert self.writer
        if isinstance(accs, list):
            self.writer.add_scalars('{}/ACC'.format(name),
                                    tag_scalar_dict=accs[0], global_step=step)
            self.writer.add_scalars('{}/AUC'.format(name),
                                    tag_scalar_dict=accs[1], global_step=step)
            self.writer.add_scalars('{}/EER'.format(name),
                                    tag_scalar_dict=accs[2], global_step=step)
            self.writer.add_scalars('{}/AP'.format(name),
                                    tag_scalar_dict=accs[3], global_step=step)
        else:
            self.writer.add_scalars('{}/AUC'.format(name),
                                    tag_scalar_dict=accs, global_step=step)

        if feas is not None:
            metadata = []
            mat = None
            for key in feas:
                for i in range(feas[key].size(0)):
                    lab = 'fake' if label[key][i] == 1 else 'real'
                    metadata.append('{}_{:02d}_{}'.format(key, int(i), lab))
                if mat is None:
                    mat = feas[key]
                else:
                    mat = torch.cat((mat, feas[key]), dim=0)

            self.writer.add_embedding(mat, metadata=metadata, label_img=None,
                                      global_step=step, tag='default')
