from model import *
from CIFAR_100_DataSet import *
import argparse
import time

import os
import datetime


parser = argparse.ArgumentParser(description='MobileNet # Classifier Training With Pytorch')
# DataSet params:
parser.add_argument('--dataset_path',  default='dataset', type=str,   help='dataset name') 
parser.add_argument('--dataset_split', default='train',   type=str,   help='train or test') 
parser.add_argument('--batch_size',    default=150,       type=int,   help='150 for single RTX 2080 TI') 
# Net:
parser.add_argument('--net_version',   default='v3',      type=str,   help='network version') 
# Optimizer params: 
parser.add_argument('--optimizer_kind',default='RMSprop', type=str,   help='optimizer type') 
parser.add_argument('--lr',            default=0.1,       type=float, help='learning rate') 
parser.add_argument('--momentum',      default=0.9,       type=float, help='momentum') 
parser.add_argument('--weight_decay',  default=1e-5,      type=float, help='weight_decay') 
parser.add_argument('--step_size',     default=3,         type=int,   help='step_size') 
# Functional HW params:
parser.add_argument('--num_gpus',      default=1,         type=int,   help='num_gpus') 
parser.add_argument('--num_epoch',     default=300,       type=int,   help='num_epoch') 
parser.add_argument('--save_freq',     default=5,         type=int,   help='checkpoint save frequency')

args = parser.parse_args()

train = CIFAR_100(path2DS = args.dataset_path, split = args.dataset_split, batch_size = args.batch_size)
print('[*] Model initialization...')
net = MobileNet(n_class=100, input_size=224, version = args.net_version)
print('[+] Done!')

if (args.num_gpus!=1)&torch.cuda.is_available():
    net = torch.nn.DataParallel(net)
print('[*] Moving to Device...')
net = net.cuda()
print('[+] Done!')

opt       = torch.optim.RMSprop if args.optimizer_kind == 'RMSprop' else torch.optim.SGD
optimizer = opt(net.parameters(),args.lr, args.momentum, args.weight_decay)
criterion = torch.nn.CrossEntropyLoss().cuda()

date_time = datetime.datetime.now()
folder_name = str(date_time.year)+'_'+str(date_time.month)+\
                '_'+str(date_time.day)+'_'+str(date_time.hour)+'_'+str(date_time.minute)
if not os.path.exists(folder_name):
    os.makedirs(os.path.join('./checkpoints',folder_name))

# if not os.path.exists('./checkpoints'):
#     os.makedirs('./checkpoints')
    
num_batches = (train.length()//args.batch_size)+1
train_batch_generator = range(num_batches)
print('[*] Start...')
total_epoch_duration = 0
for epoch in range(args.num_epoch):
    print('    [*] Epoch #{}'.format(epoch))
    train.reset_indexes()
    # learning  rate  decay  rate  of  0.01  every  args.step_size  epochs.
    if (epoch != 0) & (epoch % args.step_size == 0):
        step = epoch // args.step_size
        lr_mod   = args.lr * 0.8 ** step
        print('    [*] Changing LR to -->',lr_mod)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_mod
    
    epoch_start = time.time()
    total_loss = 0
    total_batch_duration = 0
    for batch_num in train_batch_generator:
        batch_start = time.time()
        batch_imgs, batch_gt = train.get_batch(batch_num)
        # Load data and put it to cuda
        batch_imgs = batch_imgs.cuda()
        batch_gt   = batch_gt.cuda()
        # Train one iteration
        optimizer.zero_grad()
        output = net(batch_imgs)
        loss = criterion(output, batch_gt)
        
        total_loss+=loss
        
        loss.backward()
        optimizer.step()
        
        batch_finish = time.time()
        batch_duration = batch_finish - batch_start
        total_batch_duration += batch_duration
        
        if (batch_num!=0)&(batch_num%int(num_batches/10)==0):
            print('        [*] Epoch = {0}  Batch = {1}\tLoss = {2:.4f}(AVG:{3:.4f})\tBatch_time = {4:.4f}(AVG:{5:.4f})'.format(
                epoch,batch_num,loss,total_loss.item()/(batch_num+1),batch_duration,total_batch_duration/(batch_num+1)))
            
    epoch_finish = time.time()
    epoch_duration = epoch_finish - epoch_start
    total_epoch_duration += epoch_duration
    print('    [+] Epoch completed in {0:.2f} sec(AVG:{1:.2f} sec/epoch)!'.format(epoch_duration, 
                                                                     total_epoch_duration/(epoch+1)))
    
    if (epoch != 0) & (epoch % args.save_freq == 0):
        checkpoint_name = './checkpoints/MobileNet{0}_{1}_{2}_{3:.4f}.pth'.format(
            args.net_version.upper(),args.optimizer_kind, epoch,total_loss.item()/batch_num)
        torch.save(net.module.state_dict(), checkpoint_name)
        print('    [*] Weights saved as {}'.format(checkpoint_name))
print('[+] Training completed!')