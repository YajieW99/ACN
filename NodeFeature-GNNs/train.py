import argparse
import os
import shutil
import time
import json
import numpy as np
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from callbacks import AverageMeter, Logger
from data_utils.data_loader_frames import VideoFolder
from utils import save_results
from loadlabels import loadlabel
from utils import weightCrossEntropyLoss
from ipdb import set_trace

parser = argparse.ArgumentParser(description='PyTorch ACN-node')

# Path related arguments

parser.add_argument('--model',
                    default='coord')
parser.add_argument('--root_frames', type=str, help='path to the folder with frames')
parser.add_argument('--json_data_train', type=str, help='path to the json file with train video meta data')
parser.add_argument('--json_data_val', type=str, help='path to the json file with validation video meta data')
parser.add_argument('--json_file_labels', type=str, help='path to the json file with ground truth labels')
# parser.add_argument('--save_path', type=str, help='frame and list save path')

parser.add_argument('--img_feature_dim', default=256, type=int, metavar='N',
                    help='intermediate feature dimension for image-based features')
parser.add_argument('--coord_feature_dim', default=256, type=int, metavar='N',
                    help='intermediate feature dimension for coord-based features')
parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                    metavar='W', help='gradient norm clipping (default: 5)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 72)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', '-l', default=10, type=int,
                    metavar='N', help='frequency to write in tensorboard (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes', default=174, type=int,
                    help='num of class in the model')
parser.add_argument('--num_boxes', default=4, type=int,
                    help='num of boxes for each image')
parser.add_argument('--num_frames', default=16, type=int,
                    help='num of frames for the model')
parser.add_argument('--dataset', default='smth_smth',
                    help='which dataset to train')
parser.add_argument('--logdir', default='./logs',
                    help='folder to output tensorboard logs')
parser.add_argument('--logname', default='exp',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')
parser.add_argument('--fine_tune', help='path with ckpt to restore')
parser.add_argument('--tracked_boxes', type=str, help='choose tracked boxes')
parser.add_argument('--shot', default=5)
parser.add_argument('--restore_i3d')
parser.add_argument('--arch',default='ck')
parser.add_argument('--restore_custom')

best_loss = 1000000


def main():
    global args, best_loss
    args = parser.parse_args()
    # create model


    if args.model == 'coord':
        from model.model_lib import VideoModelCoord as VideoModel


    model = VideoModel(args)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    if args.start_epoch is None:
        args.start_epoch = 0


    model = model.cuda()
    # model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # create training and validation dataset
    dataset_train = VideoFolder(root=args.root_frames,
                                num_boxes=args.num_boxes,
                                file_input=args.json_data_train,
                                file_labels=args.json_file_labels,
                                # frames_duration=args.num_frames,
                                args=args,
                                is_val=False,
                                if_augment=True,
                                model=args.model,
                                #save_path=args.save_path
                                )
    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              # frames_duration=args.num_frames,
                              args=args,
                              is_val=True,
                              if_augment=True,
                              model=args.model,
                              #save_path=args.save_path
                              )

    # create training and validation loader
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, drop_last=False,
        batch_size=args.batch_size , shuffle=False,
        num_workers=4, pin_memory=True
    )

    optimizer = torch.optim.SGD(model.parameters(), momentum=args.momentum,
                                lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # training, start a logger
    tb_logdir = os.path.join(args.logdir, args.logname)
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        isval = False

        # train for one epoch
        train(train_loader, model, optimizer, epoch, criterion, tb_logger)

        # # evaluate on validation set
        if (epoch + 1) % 1 == 0:

            loss = validate(val_loader, model, criterion, epoch=epoch, tb_logger=tb_logger)
        else:
            loss = 100

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
            },
            is_best,
            os.path.join(args.ckpt, args.arch.lower() + '_{}'.format(args.logname) + '_e{}'.format(epoch)))


def train(train_loader, model, optimizer, epoch, criterion, tb_logger=None):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (global_img_tensors, box_tensors, box_categories, hand_id, video_label) in enumerate(train_loader):
        # print(video_label)

        video_label_cuda = video_label.cuda()
        model.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        cls, label = model(global_img_tensors, box_categories, box_tensors, hand_id, video_label_cuda)

        # print(count30)
        cls = cls.view((-1, len(train_loader.dataset.classes)))
        device = cls.device
        loss = criterion(cls, label).cuda(device=device)

        acc1, acc5 = accuracy(cls.cpu(), label.cpu())
        # measure accuracy and record loss
        losses.update(loss.item(), hand_id.size(0))
        acc_top1.update(acc1.item(), hand_id.size(0))
        acc_top5.update(acc5.item(), hand_id.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(train_loader):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc_top1=acc_top1, acc_top5=acc_top5))
        # log training data into tensorboard
        if tb_logger is not None and i % args.log_freq == 0:
            logs = OrderedDict()
            logs['Train_IterLoss'] = losses.val
            logs['Train_Acc@1'] = acc_top1.val
            logs['Train_Acc@5'] = acc_top5.val
            # how many iterations we have trained
            iter_count = epoch * len(train_loader) + i
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, iter_count)

            tb_logger.flush()


def validate(val_loader, model, criterion, epoch=None, tb_logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    logits_matrix = []
    targets_list = []

    # switch to evaluate mode
    # isval = True
    model.eval()
    end = time.time()
  
    for i, (global_img_tensors, box_tensors, box_categories, hand_id, video_label) in enumerate(val_loader):
        # compute output
        # isval = True

        video_label_cuda = video_label.cuda()
        batchsize = hand_id.shape[0]
        with torch.no_grad():
            cls, label = model(global_img_tensors, box_categories, box_tensors, hand_id, video_label_cuda)
            cls = cls.view((-1, len(val_loader.dataset.classes)))

            # device = cls.device
            loss = criterion(cls, label)
            acc1, acc5 = accuracy(cls.cpu(), label.cpu(), topk=(1, 5))
            if args.evaluate:
                logits_matrix.append(cls.cpu().data.numpy())
                targets_list.append(label.cpu().numpy())

        # measure accuracy and record loss
        losses.update(loss.item(), batchsize)
        acc_top1.update(acc1.item(),batchsize)
        acc_top5.update(acc5.item(), batchsize)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_e {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc_top1=acc_top1, acc_top5=acc_top5))

    if args.evaluate:
        logits_matrix = np.concatenate(logits_matrix)
        targets_list = np.concatenate(targets_list)

        save_results(logits_matrix,targets_list, epoch, args)

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs['Val_EpochLoss'] = losses.avg
        logs['Val_EpochAcc@1'] = acc_top1.avg
        logs['Val_EpochAcc@5'] = acc_top5.avg
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

        tb_logger.flush()

    return losses.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


if __name__ == '__main__':
    main()
