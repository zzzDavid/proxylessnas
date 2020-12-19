import os
import os.path as osp

import argparse
import time
from tqdm import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets

from proxyless_nas.utils import AverageMeter, accuracy
from proxyless_nas import model_zoo

from pytorch2caffe.converter import pytorch2caffe_converter

model_names = sorted(name for name in model_zoo.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(model_zoo.__dict__[name]))

# setup arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    help='The path of imagenet',
    type=str,
    default='/dataset/imagenet'
)
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='0'
)
parser.add_argument(
    '-a',
    '--arch',
    metavar='ARCH',
    default='proxyless_mobile_14',
    choices=model_names,
    help='model architecture: ' +
    ' | '.join(model_names) +
    ' (default: proxyless_mobile_14)'
)
parser.add_argument(
    '-s',
    '--source',
    help='Calibration dataset image list',
    type=str,
    default='/dataset/imagenet_calib/calibration.txt'
)
parser.add_argument(
    '-r',
    '--root',
    help='Calibration dataset image directory',
    type=str,
    default='/dataset/imagenet_calib/img/'
)
parser.add_argument(
    '-d',
    '--dimension',
    help='input dimension',
    type=int,
    default=224
)

args = parser.parse_args()
# set gpu devices
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# setup data loader
data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        osp.join(
            args.path,
            "val"),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406],
                    std=[
                        0.229,
                        0.224,
                        0.225]),
            ])),
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

device = torch.device('cuda:0')
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().to(device)

# set dummy input
input_tensor = torch.ones([1, 3, args.dimension, args.dimension])



def eval():
    net = model_zoo.__dict__[args.arch](pretrained=True)
    # get flops
    flops = int(net.get_flops(input_tensor)[0])
    # get #parameters
    param = sum([tensor.numel() for tensor in net.parameters()])

    net = torch.nn.DataParallel(net).to(device)
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc='Test') as t:
            for i, (_input, target) in enumerate(data_loader):
                target = target.to(device)
                _input = _input.to(device)

                # compute output
                output = net(_input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), _input.size(0))
                top1.update(acc1[0].item(), _input.size(0))
                top5.update(acc5[0].item(), _input.size(0))

                t.set_postfix({
                    'Loss': losses.avg,
                    'Top1': top1.avg,
                    'Top5': top5.avg
                })
                t.update(1)

    print('Loss:', losses.avg, '\t Top1:', top1.avg, '\t Top5:', top5.avg)
    print(f"FLOPs = {flops}" + "\n" + f"#params = {param}") 

def export_caffe():
    net = model_zoo.__dict__[args.arch](pretrained=True)
    net_name = args.arch.replace('_', '')
    caffe_result = 'caffe_nets'
    if not osp.exists(caffe_result):
        os.makedirs(caffe_result)
    print(net)
    t2c = pytorch2caffe_converter(net)
    t2c.set_input(input_tensor, args.source, args.root, 1, args.dimension, args.dimension)
    t2c.trans_net(net_name)
    t2c.save_prototxt(os.path.join(caffe_result, net_name + '.prototxt'))
    t2c.save_caffemodel(os.path.join(caffe_result, net_name + '.caffemodel'))
    print('generated caffe model : ' + net_name)
    

if __name__ == '__main__':
    eval()
    export_caffe()
