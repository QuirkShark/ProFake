import argparse
from cmath import rect
# from ast import main
from email.policy import strict
from tkinter.tix import MAIN
from unicodedata import name
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import torch.nn as nn
import os
import torch.multiprocessing as mp
import random
from tqdm import tqdm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.distributed as dist
import cv2
from PIL import Image
from trainer.whole_model import Model
from dataprocess.FF_data_deg import FaceForensicsDataset, DFDCDataset, FaceShifterDataset, CelebDFDataset

from utils.metrics import Metrics
from sklearn.metrics import recall_score
import dlib
from torchvision import transforms as T

dset = ['ALL','FF_DF', 'FF_NT', 'FF_FS', 'FF_F2F']
# Add plot for tensorboard
detector = dlib.get_frontal_face_detector()
def crop_face(img,rect,res):
    t = max(int(rect[0].top()*0.9),0)
    b = min(int(rect[0].bottom()*1.1),img.shape[0])
    l = max(int(rect[0].left()*0.9),0)
    r = min(int(rect[0].right()*1.1),img.shape[1])
    img = img[t:b,l:r]
    img = cv2.resize(img,(res,res),interpolation=cv2.INTER_CUBIC)
    return img

def get_accracy(output, label):
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)
    return accuracy

def get_prediction(output, label):
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    datas = torch.cat((prob, label.float()), dim=1)
    return datas

def test_epoch(model, test_data_loaders, step):
    # --------------eval------------
    model.setEval()

    def run(data_loader, name):
        statistic = None
        metric = Metrics()
        metric_4 = Metrics()
        losses = []
        losses_4 = []
        acces = []
        acces_4 = []
        print('Len of data_loader=', len(data_loader))
        # test ckpt in whole test set

        for i, batch in tqdm(enumerate(data_loader)):
            img, label = batch
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            score, latent, loss = model.inference(img, label)
            label_2, label_4, label_pred = score
            loss_cls, loss_4 = loss
            tmp_data = get_prediction(label_2.detach(), label)
            losses.append(loss_cls.cpu().detach().numpy())
            losses_4.append(loss_4.cpu().detach().numpy())
            acces.append(get_accracy(label_2, label))
            acces_4.append(get_accracy(label_pred, label))
            metric.update(label.detach(), label_2.detach())
            metric_4.update(label.detach(),label_pred.detach())
        model.update_tensorboard(None, step, acc=None, datas=statistic, name='test/{}'.format(name))
        avg_loss_4 = np.mean(np.array(losses_4))
        avg_loss = np.mean(np.array(losses))
        info = "|Test Loss {:.4f} {:.4f}".format(avg_loss,avg_loss_4)
        mm = metric.get_mean_metrics()
        mm_4 = metric_4.get_mean_metrics()
        mm_str = ""
        mm_str += "\t|Acc {:.4f} (~{:.2f})".format(mm[0], mm[1])
        mm_str += "\t|AUC {:.4f} (~{:.2f})".format(mm[2], mm[3])
        mm_str += "\t|EER {:.4f} (~{:.2f})".format(mm[4], mm[5])
        mm_str += "\t|AP {:.4f} (~{:.2f})".format(mm[6], mm[7])
        info += mm_str
        mm_str = ""
        mm_str += "\t|Acc_4 {:.4f} (~{:.2f})".format(mm_4[0], mm_4[1])
        mm_str += "\t|AUC_4 {:.4f} (~{:.2f})".format(mm_4[2], mm_4[3])
        mm_str += "\t|EER_4 {:.4f} (~{:.2f})".format(mm_4[4], mm_4[5])
        mm_str += "\t|AP_4 {:.4f} (~{:.2f})".format(mm_4[6], mm_4[7])
        info += mm_str
        print(info)
        metric.clear()
        metric_4.clear()
        return (mm[0], mm[2], mm[4], mm[6])

    keys = test_data_loaders.keys()
    datas = [{}, {}, {}, {}]
    for i, key in enumerate(keys):
        print('[{}/{}]Testing from {} ...'.format(i+1, len(keys), key))
        dataloader = test_data_loaders[key]
        ret = run(dataloader, key)
        for j, data in enumerate(ret):
            datas[j][key] = data
    model.update_tensorboard_test_accs(datas, step, feas=None)




def train_epoch(gpu, model, train_data_loader, epoch, cur_acc, savedir,test_data_loaders=None, dataset='FF_DF'):

    if gpu== 0:
        print("===> Epoch[{}] start!".format(epoch))
        
    best_acc = cur_acc
    model.setTrain()
    # --------------train------------
    eval_step = len(train_data_loader) // 10    # eval 10 times per epoch
    step_cnt = epoch * len(train_data_loader)
    losses = []
    l2_losses = []
    lh_losses = []
    ll_losses = []
    hh_losses = []
    lll_losses = []
    acces_hh = []
    acces_ll = []
    acces = []
    acces_lq = []
    acces_hq = []
    com_list = []
    feat_list = []
    for iteration, batch in tqdm(enumerate(train_data_loader)):
        model.setTrain()
        img, label = batch
        img = img.cuda(non_blocking=True)
        with torch.autograd.set_detect_anomaly(True):
            gt, pred, ret = model.optimize(img,label,epoch)
            label,label_hq,label_lq = gt
            label_2, label_hq_,label_lq_ = pred
            loss,loss_l2,loss_lh,loss_ll = ret
        
        
        losses.append(loss.cpu().detach().numpy())
        l2_losses.append(loss_l2.cpu().detach().numpy())
        ll_losses.append(loss_ll.cpu().detach().numpy())
        lh_losses.append(loss_lh.cpu().detach().numpy())
        
        
        acces_lq.append(get_accracy(label_lq_, label_lq),)
        acces_hq.append(get_accracy(label_hq_, label_hq),)
        
        
        acces.append(get_accracy(label_2, label),)

        ### desc
        if iteration%100==0:
            print(f"[TRAIN] Iter: {iteration} \
            Loss: {np.mean(np.array(losses))} \
            Loss_l2: {np.mean(np.array(l2_losses))} \
            Loss_ll: {np.mean(np.array(ll_losses))} \
            Loss_lh: {np.mean(np.array(lh_losses))} \
            Acc:{np.mean(np.array(acces))}\
            Acc_hq:{np.mean(np.array(acces_hq))} \
            Acc_lq:{np.mean(np.array(acces_lq))}")
        



        if iteration % 500 == 0 and gpu == 0:
            info = "[{}/{}]\n".format(iteration, len(train_data_loader))
            avg_loss = np.mean(np.array(losses))
            info += "\tLoss Cls:{:.4f}\n".format(avg_loss)
            avg_acc = np.mean(np.array(acces))
            info += '\tAVG Acc\t{:.4f}'.format(avg_acc)
            acces.clear()
            losses.clear()
            
            model.update_tensorboard(avg_loss, step_cnt, acc=avg_acc, name='train')
            


        if (step_cnt+1) % eval_step == 0 and gpu == 0:
            if test_data_loaders is not None:
                test_epoch(model, test_data_loaders, step_cnt)
            model.update_tensorboard(avg_loss, step_cnt, acc=avg_acc, name='train')
            model.save_ckpt(dataset, epoch, iteration, savedir, best=False)
        step_cnt += 1
    if gpu == 0:
        print('Saving the model......')
        model.save_ckpt(dataset, epoch, iteration, savedir, best=False)
    return best_acc


def train(gpu,args):
    print('start training')
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed) 
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    logdir = "{}/train".format(args.logdir)
    
    
    
    # load your model
    model = Model(args, logdir = logdir, train = True)
    torch.cuda.set_device(gpu)
    model.classifier.cuda(gpu)
    model.synthesizer.cuda(gpu)
    model.backbone.cuda(gpu)
    



    # restore ckpts
    if args.backbone_w is not None:
        weights_dict = torch.load(args.backbone_w,map_location='cpu')["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print("Load backbone pretrained weights", args.backbone_w)
        model.backbone.load_state_dict(weights_dict,strict = False)
    if args.classifier_path is not None:
        model.load_ckpt(args.classifier_path,args.synthesizer_path,args.backbone_path)
    
    # for linear projection on backbone
    if args.freeze_layers:
        for name, para in model.backbone.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    



    
    model.classifier = nn.parallel.DistributedDataParallel(model.classifier, device_ids=[gpu],find_unused_parameters = False)
    model.synthesizer = nn.parallel.DistributedDataParallel(model.synthesizer, device_ids=[gpu],find_unused_parameters=False)
    model.backbone = nn.parallel.DistributedDataParallel(model.backbone, device_ids=[gpu],find_unused_parameters=False)
    model.cls_criterion = model.cls_criterion.cuda(gpu)
    model.l1loss = model.l1loss.cuda(gpu)


    # -----------------load dataset--------------------------
    train_set = FaceForensicsDataset(dataset=args.dset, mode='train', res=args.resolution, train=True,quality = 'c23')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,num_replicas=args.world_size,rank=rank)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batchSize,
                                                    shuffle=False, num_workers=8, pin_memory=True,
                                                    sampler=train_sampler)
    TESTLIST = {
        'FF_DF': "non-input",
        'FF_NT': "non-input",
        'FF_FS': "non-input",
        'FF_F2F': "non-input",
    }
    def get_data_loader(name):
        # -----------------load dataset--------------------------
        test_set = FaceForensicsDataset(dataset=name, mode='test', res=args.resolution, train=False,quality='c40')
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batchSize,
                                                       shuffle=False, num_workers=int(args.workers))
        return test_data_loader
    test_data_loaders = {}
    for list_key in TESTLIST.keys():
        test_data_loaders[list_key] = get_data_loader(list_key)
    
    # ----------------Train by epochs--------------------------
    best_acc = 0
    dataset = 'FF++'
    
    



    if gpu== 0:
        model.define_summary_writer()

    tb_writer = model.writer
    init_img = torch.zeros((1,3,256,256),device=torch.device("cuda"))
    tb_writer.add_graph(model.backbone,init_img)

    # add your model graph to tensorboard
    init_img = torch.zeros((1,3,256,256),device=gpu)
    model.writer.add_graph(model.backbone,init_img)


    for epoch in range(args.start_epoch, args.nEpochs + 1):
        best_acc = train_epoch(gpu, model, train_data_loader, epoch, best_acc, args.savedir, test_data_loaders=test_data_loaders,
        dataset=dataset)
        print("===> Epoch[{}] end with acc {:.4f}!".format(epoch, best_acc))
    print("Stop Training on best validation accracy {:.4f}".format(best_acc))
    dist.destroy_process_group()




def main():
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    opt = parser.parse_args()
    opt.world_size = opt.gpus * opt.nodes                #
    os.environ['MASTER_ADDR'] = ip                       #
    os.environ['MASTER_PORT'] = opt.masterport           #
    mp.spawn(train, nprocs=opt.gpus, args=(opt,))        #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model setting
    parser.add_argument('--meta', default="init",
                    type=str, help='the feature space')
    parser.add_argument("--pretrained", default=None, type=str,
                    help="path to pretrained model (default: none)")
    parser.add_argument("--synthesis",type = str, default=None)
    parser.add_argument('--cuda',default=True, action='store_true', help='enable cuda')
    parser.add_argument('--backbone',default='resnet50')
    parser.add_argument('--backbone_w',default='./weights/convnext_base_1k_224_ema.pth')
    parser.add_argument('--quality',default = 'raw')

    # training weights
    parser.add_argument("--classifier_path",type = str, default=None)
    parser.add_argument("--synthesizer_path",type = str, default=None)
    parser.add_argument("--backbone_path",type = str, default=None)


    parser.add_argument('--freeze-layers', type=bool, default=False)
    # exp settings
    parser.add_argument('--logdir', default='/data/code/deepfake_detection/Robust_Deepfake_Detection/logs',
                    help='folder to output images')
    parser.add_argument('--savedir', default='./saved',
                    help='folder to output images')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    # dataset
    parser.add_argument('--dset', type=str, choices=dset,
                    help='method in FF++')
    # training settings
    parser.add_argument('--train_batchSize', type=int,
                    default=24, help='input batch size')
    parser.add_argument('--eval_batchSize', type=int,
                    default=24, help='eval batch size')
    parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=8)
    parser.add_argument('--resolution', type=int, default=256,
                    help='the resolution of the output image to network')
    parser.add_argument('--test_batchSize', type=int,
                    default=128, help='test batch size')
    parser.add_argument('--dataname', type=str, help='dataname')
    
    parser.add_argument('--save_epoch', type=int, default=1,
                    help='the interval epochs for saving models')
    parser.add_argument('--rec_iter', type=int, default=100,
                    help='the interval iterations for recording')
    parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
    parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum, Default: 0.9")
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                    help="Momentum, Default: 0.0005")
    parser.add_argument('--wd', type=float, default=5e-2,
                    help='weight decay for backbone')
    parser.add_argument("--nEpochs", type=int, default=30,
                    help="number of epochs to train for")
    parser.add_argument("--start_epoch", default=0, type=int,
                    help="Manual epoch number (useful on restarts)")
    # for distributed parallel 
    parser.add_argument('-n', '--nodes', default=1,
                    type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
    parser.add_argument('-mp', '--masterport', default='8888', type=str,
                    help='ranking within the nodes')
    parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
    
    

    main()