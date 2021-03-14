from model import *
from CIFAR_100_DataSet import *
import argparse
import time
from meter import *
import sys
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='MobileNet # Classifier Training With Pytorch')
# DataSet params:
parser.add_argument('--dataset_path',  default='dataset', type=str,   help='dataset name') 
parser.add_argument('--dataset_split', default='test',    type=str,   help='train or test') 
parser.add_argument('--batch_size',    default=6,       type=int,   help='150 for single RTX 2080 TI') 
# Net:
parser.add_argument('--net_version',   default='v3',      type=str,   help='network version') 
# Functional HW params:
parser.add_argument('--num_gpus',      default=1,         type=int,   help='num_gpus') 
# Checkpoint
parser.add_argument('--checkpoint',    default='',        type=str,   help='checkpoint name for test') 

args = parser.parse_args()
if args.checkpoint == '':
    print('[!] No checkpoint for downloading! You can find checkpoint in ./checkpoints folder')
    for file in os.listdir('./checkpoints'):
        print(file)
    sys.exit(0)
    
test = CIFAR_100(path2DS = args.dataset_path, split = args.dataset_split, batch_size = args.batch_size)
print('[*] Model initialization...')
net = MobileNet(n_class=100, input_size=224, version = args.net_version)
print('[*] Loading model weights ...')
stat_dict = torch.load(os.path.join('./checkpoints',args.checkpoint), map_location='cpu')
net.load_state_dict(stat_dict)
print('[+] Done!')

if (args.num_gpus!=1)&torch.cuda.is_available():
    net = torch.nn.DataParallel(net)
print('[*] Moving to Device...')
net = net.cuda()
net.eval()
print('[+] Done!')

correct_meter = AverageMeter()
global_start  = time.time()

pred_raw_all   = []
pred_prob_all  = []
pred_label_all = []

with torch.no_grad():
    num_batches = (test.length()//args.batch_size)+1
    test_batch_generator = range(num_batches)
    
    for batch_num in tqdm(test_batch_generator):
        batch_start = time.time()
        
        batch_imgs, batch_gt = test.get_batch(batch_num)
        batch_imgs = batch_imgs.cuda()
        batch_gt   = batch_gt.cuda()
        
        output = net(batch_imgs)
        _, preds = torch.max(output, dim=1)
        correct_ = preds.eq(batch_gt).sum().item()
        correct_meter.update(correct_, 1)
        
    accuracy = correct_meter.sum / test.length()

    total_time = time.time() - global_start
    print('total_time = {:.3f}'.format(total_time))
print('[+] Accuracy = ',accuracy)