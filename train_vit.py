from __future__ import print_function
import json
import glob
from itertools import chain
import os
import random
import zipfile
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm
from vit_pytorch import ViT
from models.big_bird_vit import BigBirdViT
torch.autograd.set_detect_anomaly(True)
import argparse
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(1)

parser = argparse.ArgumentParser('ViT Args', add_help=False)

parser.add_argument('--train_batch_size', default=16, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--image_size', default=368, type=int)
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--vit_arch', default="OriginalViT", type=str,help="OriginalViT, BigBirdViT, GraphViT")



#Training specs
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.7, type=float)

## GPU ARGS ##
parser.add_argument('--use_gpu', default=True, type=bool)
parser.add_argument('--distributed', default=True, type=bool)
parser.add_argument('--local_rank', default=1, type=int)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--resume', default=False)
parser.add_argument('--tensorboard_dir', default="tensorboard")
parser.add_argument('--training_dir', default="training")
parser.add_argument('--resume_ckpt')


args = parser.parse_args()

def main(args):
    LOCAL_RANK = args.local_rank
    DIST = args.distributed
    DIST = True
    BASE_CKPT_PATH = "/home/vp.shivasan/IvT/SparseAttentionViT/ckpts/"
    MODEL_CKPT_PATH = "/home/vp.shivasan/IvT/SparseAttentionViT/ckpts/" + args.vit_arch +"/"
    BASE_TENSORBOARD_PATH = "/home/vp.shivasan/IvT/SparseAttentionViT/tensorboard/"
    BASE_LOG_NAME = str(args.image_size)+ "_p" +str(args.patch_size)
    CUSTOM_LOG_NAME = "_BB_Global-200epochs_1e-3" # Change this
    RUN_LOG_NAME = BASE_LOG_NAME + CUSTOM_LOG_NAME
    RESUME = False
    resume_ckpt_path = ""


    if(LOCAL_RANK == 1):
        RUN_CKPT_SAVE_PATH = BASE_CKPT_PATH + RUN_LOG_NAME + "/"
        RUN_TENSORBOARD_PATH = BASE_TENSORBOARD_PATH + RUN_LOG_NAME + "/"
        if not os.path.exists(RUN_CKPT_SAVE_PATH):
            print("Creating model checkpoint dir")
            os.makedirs(RUN_CKPT_SAVE_PATH)
        else:
            print("Model checkpoint dir exsist, make sure log name is different or else it will get overwritten")
        
        if not os.path.exists(RUN_TENSORBOARD_PATH):
            print("Creating tensorboard")
            os.makedirs(RUN_TENSORBOARD_PATH)
        else:
            print("tensorboard dir exisist")
        
        writer = SummaryWriter(RUN_TENSORBOARD_PATH)

    if DIST:
        print("Intiliasing distributed process")
        DEVICE = torch.device('cuda:%d' % LOCAL_RANK)
        torch.cuda.set_device(LOCAL_RANK)
        print(DEVICE)
        dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.num_gpus, rank=LOCAL_RANK)
        print("Distributed process running")
    
    else:
        CUDA_ = 'cuda:1'
        DEVICE = torch.device(CUDA_)


    # BATCH_SIZE_TRAIN = 64
    # BATCH_SIZE_TEST = 128
    # IMAGE_SIZE = 368
    transform_train = torchvision.transforms.Compose([torchvision.transforms.Resize((args.image_size,args.image_size)),
                                                        transforms.RandomResizedCrop(args.image_size),
                                                        transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize((args.image_size,args.image_size)),
                                                        transforms.CenterCrop(args.image_size),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    train_set = torchvision.datasets.ImageFolder(root="/home/vp.shivasan/IvT/data/imagenette2/train",transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(root="/home/vp.shivasan/IvT/data/imagenette2/val",transform=transform_test)
    

    if(DIST):
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_set)
        sampler_val = torch.utils.data.distributed.DistributedSampler(test_set,shuffle = False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(test_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.train_batch_size, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(
        sampler_val, args.test_batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=batch_sampler_train,num_workers=2)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_sampler=batch_sampler_val,num_workers=2)
    
    

    if(args.vit_arch == "OriginalViT"):
        model = ViT(
            image_size = args.image_size,
            patch_size = args.patch_size,
            num_classes = 10,
            dim = 512,
            depth = 3,
            heads = 8,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1,
            pool = 'mean'
        )

    elif(args.vit_arch == "BigBirdViT"):
        attentions_to_use = ["Global"]

        model = BigBirdViT(
            image_size = args.image_size,
            patch_size = args.patch_size,
            num_classes = 10,
            dim = 512,
            depth = 3,
            heads = 8,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1,
            attention_to_use = attentions_to_use,
            pool = 'mean'
        )
    else:
        print("Error Unknown Model ",args.vit_arch)
        exit()
    

    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True 
    

    model = model.to(DEVICE)
    if DIST:
        model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[LOCAL_RANK])                       
    else:
        model = nn.DataParallel(model,device_ids=[1,]).to('cuda:1')
    
    
    if(RESUME):
        model.load_state_dict(torch.load(resume_ckpt_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', verbose=True)

    best_val_acc = 0.0
    print("Training ", args.vit_arch)
    for epoch in tqdm(range(args.epochs)):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
        if(LOCAL_RANK == 1):
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )
            if(best_val_acc < epoch_val_accuracy):
                torch.save(model.state_dict(),  RUN_CKPT_SAVE_PATH + "best_ckpt.pt")
                my_dict = {'Epoch':epoch+1,'Val_acc':epoch_val_accuracy.item()}
                with open(RUN_CKPT_SAVE_PATH + "meta_json.json", "w") as fp:
                    json.dump(my_dict,fp) 
                best_val_acc = epoch_val_accuracy
            writer.add_scalar('Train loss', epoch_loss, epoch)
            writer.add_scalar('Train acc', epoch_accuracy, epoch)
            writer.add_scalar('Val loss', epoch_val_loss, epoch)
            writer.add_scalar('Val acc', epoch_val_accuracy, epoch)
            writer.add_scalar('LR', [ group['lr'] for group in optimizer.param_groups ][0], epoch)
        scheduler.step(epoch_val_loss)

    if(LOCAL_RANK == 1):
        torch.save(model.state_dict(), RUN_CKPT_SAVE_PATH + "best_ckpt_" + str(epoch)+".pt")

if __name__ == '__main__':
  main(args)

# CLassic: python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 train_vit.py
# BigBird: python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 train_vit.py --vit_arch BigBirdViT
# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 train_vit.py --vit_arch BigBirdViT --patch_size=8 --image_size=184

# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 --master_port 47770 train_vit.py --vit_arch OriginalViT --patch_size=8 --image_size=384
# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 --master_port 47770 train_vit.py --vit_arch BigBirdViT --patch_size=8 --image_size=384