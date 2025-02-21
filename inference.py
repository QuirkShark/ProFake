"""
eval pretained model.
"""
import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from dataprocess.FFdata import FaceForensicsDataset
from dataprocess.OSN import OSNDataset as OSN
from trainer.whole_model import Model 
from utils.metrics import Metrics
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
sys.path.append('/data/code/wyp/PyDeepFakeDet/preprocess')


dset = ['ALL','FF_DF','FF_F2F','FF_NT','FF_FS']
# os.environ['CUDA_VISIBLE_DEVICES']='2'
parser = argparse.ArgumentParser()

# model setting
parser.add_argument('--meta', default="init",
                    type=str, help='the feature space')

# dataset
parser.add_argument('--dset', type=str, choices=dset,
                    help='method in FF++')
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

# setting
parser.add_argument('--save_epoch', type=int, default=1,
                    help='the interval epochs for saving models')
parser.add_argument('--rec_iter', type=int, default=100,
                    help='the interval iterations for recording')

# trainning config
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum, Default: 0.9")
parser.add_argument("--weight_decay", default=0.0005, type=float,
                    help="Momentum, Default: 0.0005")

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

parser.add_argument('--logdir', default='/data/code/deepfake_detection/logs',
                    help='folder to output images')
parser.add_argument('--savedir', default='./saved',
                    help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--pretrained", default=None, type=str,
                    help="path to pretrained model (default: none)")
parser.add_argument('--cuda',default=True, action='store_true', help='enable cuda')
# Load your weights
parser.add_argument("--classifier_path",type = str, default='/data/code/deepfake_detection/Robust_Deepfake_Detection/saved/whole-model/ALL/FF++_old/epoch_1_iter_16109_classifier.pth')
parser.add_argument("--synthesizer_path",type = str, default='/data/code/deepfake_detection/Robust_Deepfake_Detection/saved/whole-model/ALL/FF++_old/epoch_1_iter_16109_syn.pth')
parser.add_argument("--backbone_path",type = str, default='/data/code/deepfake_detection/Robust_Deepfake_Detection/saved/whole-model/ALL/FF++_old/epoch_1_iter_16109_backbone.pth')
parser.add_argument('--wd', type=float, default=5e-2,
                    help='weight decay for backbone')
# parser.add_argument('--backbone_w',default='./weights/convnext_base_1k_224_ema.pth')
parser.add_argument('--backbone_w',default='/data/code/deepfake_detection/Robust_Deepfake_Detection/weights/convnext_base_1k_224_ema.pth')
parser.add_argument('--quality',type=str,default='raw')
parser.add_argument("--low", dest="low",type=str)
parser.add_argument("--method", dest="method", help="The name of dataset", type=str)
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


def analysis_feature(gpu,args):
    print('start analysising your feature')
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
    logdir = "{}/test".format(args.logdir) 
    model = Model(args, logdir = logdir, train = True)
    torch.cuda.set_device(gpu)
    model.classifier.cuda(gpu)
    model.synthesizer.cuda(gpu)
    model.backbone.cuda(gpu)
    if args.backbone_w is not None:
        weights_dict = torch.load(args.backbone_w,map_location='cpu')["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # print(model.backbone.load_state_dict(weights_dict,strict = False))
        print("Load backbone pretrained weights", args.backbone_w)
        model.backbone.load_state_dict(weights_dict,strict = False)
    if args.classifier_path is not None:
        model.load_ckpt(args.classifier_path,args.synthesizer_path,args.backbone_path)
    model.classifier = nn.parallel.DistributedDataParallel(model.classifier, device_ids=[gpu],find_unused_parameters = True)
    model.synthesizer = nn.parallel.DistributedDataParallel(model.synthesizer, device_ids=[gpu],find_unused_parameters=True)
    model.backbone = nn.parallel.DistributedDataParallel(model.backbone, device_ids=[gpu],find_unused_parameters=True)
    model.cls_criterion = model.cls_criterion.cuda(gpu)
    model.l1loss = model.l1loss.cuda(gpu)
    model.cls_criterion = model.cls_criterion.cuda(gpu) # criterion to gpu
    model.l1loss = model.l1loss.cuda(gpu)
    batch_size=args.train_batchSize
    TESTLIST = {
        'FF_DF': "FF_DF",
        'FF_F2F':"FF_F2F",
        'FF_FS':"FF_FS",
        'FF_NT':"FF_NT"
        # 'OSN': "OSN"
    }
    def get_data_loader(name):
        # -----------------load dataset--------------------------
        
        # load for FF++
        test_set = FaceForensicsDataset(dataset=name, mode='test', res=args.resolution, train=False,quality=args.quality)
        
        
        # load for OSN
        # test_set = OSN()
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batchSize,
                                                       shuffle=False, num_workers=int(args.workers))
        return test_data_loader
    test_data_loaders = {}
    for list_key in TESTLIST.keys():
        test_data_loaders[list_key] = get_data_loader(list_key)

    # -----------------Test dataset--------------------------
    keys = test_data_loaders.keys()
    model.setEval()
    for i, key in enumerate(keys):
        metric = Metrics()
        metric_4 = Metrics()
        feature_similarity = []
        
        print('[{}/{}]Testing from {} ...'.format(i+1, len(keys), key))
        data_loader = test_data_loaders[key]
        print('Len of data_loader=', len(data_loader))
        for i, batch in tqdm(enumerate(data_loader)):
            img, label = batch
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            similarity, label_hq, label_lq = model.analysis(img,label,1)
            feature_similarity.append(similarity)


def inference(gpu,args):
    print('start inferencing')
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

    # logdir = "{}/test".format(args.logdir)
    logdir = '/data/code/deepfake_detection/Robust_Deepfake_Detection/logs'
    # ------------------- speed up
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
        # print(model.backbone.load_state_dict(weights_dict,strict = False))
        print("Load backbone pretrained weights", args.backbone_w)
        model.backbone.load_state_dict(weights_dict,strict = False)
    if args.classifier_path is not None:
        model.load_ckpt(args.classifier_path,args.synthesizer_path,args.backbone_path)

    model.classifier = nn.parallel.DistributedDataParallel(model.classifier, device_ids=[gpu],find_unused_parameters = True)
    model.synthesizer = nn.parallel.DistributedDataParallel(model.synthesizer, device_ids=[gpu],find_unused_parameters=True)
    model.backbone = nn.parallel.DistributedDataParallel(model.backbone, device_ids=[gpu],find_unused_parameters=True)
    model.cls_criterion = model.cls_criterion.cuda(gpu)
    model.l1loss = model.l1loss.cuda(gpu)


    # model.model = nn.parallel.DistributedDataParallel(model.model, device_ids=[gpu],
    #                                                   find_unused_parameters = True) # model to gpu
    model.cls_criterion = model.cls_criterion.cuda(gpu) # criterion to gpu
    model.l1loss = model.l1loss.cuda(gpu)
    batch_size=args.train_batchSize
    TESTLIST = {
        # 'FF_DF': "FF_DF",
        # 'FF_F2F':"FF_F2F",
        # 'FF_FS':"FF_FS",
        # 'FF_NT':"FF_NT",
        # 'OSN': "OSN",
        'LIVE': "LIVE",
        # 'FFIW':"FFIW"

    }
    def get_data_loader(name):
        # -----------------load dataset--------------------------
        test_set = FaceForensicsDataset(dataset=name, mode='test', res=args.resolution, train=False,quality = args.quality)
        # test_set = OSN()
        # test_set = FFIWDataset(args.low)
        # test_set = LIVEDataset(method=args.method)
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batchSize,
                                                       shuffle=False, num_workers=int(args.workers))
        return test_data_loader
    test_data_loaders = {}
    for list_key in TESTLIST.keys():
        test_data_loaders[list_key] = get_data_loader(list_key)

    # -----------------Test dataset--------------------------
    keys = test_data_loaders.keys()
    model.setEval()
    for i, key in enumerate(keys):
        metric = Metrics()
        metric_4 = Metrics()
        losses = []
        losses_4 = []
        acces = []
        acces_4 = []
        print('[{}/{}]Testing from {} ...'.format(i+1, len(keys), key))
        data_loader = test_data_loaders[key]
        print('Len of data_loader=', len(data_loader))
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

            # if tsne_mode == True:




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
    dist.destroy_process_group()
    return
def main():
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    opt = parser.parse_args()
    print(opt)
    opt.world_size = opt.gpus * opt.nodes                #
    os.environ['MASTER_ADDR'] = ip                       #
    os.environ['MASTER_PORT'] = opt.masterport           #
    mp.spawn(inference, nprocs=opt.gpus, args=(opt,))        #

if __name__ == '__main__':
    main()
