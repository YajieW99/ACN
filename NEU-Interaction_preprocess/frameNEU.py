import torch
from data_utils.data_loader_16_frames import Video10Extract
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse

parser = argparse.ArgumentParser(description='Produce NEU-I frames')
parser.add_argument('--root', default='./NEU-Interaction/video/')
parser.add_argument('--json_file_input', default='./NEU-Interaction/validation.json')
parser.add_argument('--json_file_labels', default='./NEU-Interaction/labels.json')
parser.add_argument('--save_path', default='./NEU-Interaction/')


def main():
    global args
    args = parser.parse_args()
    print('VideoFolder the dataset_train')
    dataset_train = Video10Extract(root=args.root,
                                   file_input=args.json_file_input,
                                   file_labels=args.json_file_labels,
                                   save_path=args.save_path)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=72, shuffle=True,
        num_workers=4, drop_last=False,
        pin_memory=True
    )
    print(len(train_loader))
    for i, index in enumerate(train_loader):    #

        print('video%d' % i, end='\r')


if __name__ == '__main__':

    main()
