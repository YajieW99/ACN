import os
from os.path import join
from torchvision import transforms
from torchvision.transforms import Compose
import numpy as np
from PIL import Image
import torch
from data_utils import gtransforms
from data_utils.data_parser import WebmDataset
import json
import skvideo.io
from tqdm import tqdm
from ipdb import set_trace


class VideoFolder(torch.utils.data.Dataset):
    """
    Something-Something dataset based on *frames* extraction
    """

    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 save_path,
                 args=None,
                 is_test=False,
                 multi_crop_test=False,
                 is_val=False,
                 num_boxes=4,
                 model=None,
                 if_augment=True,
                 crop=[80, 80]):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        :param *crop: crop and resize the box's rgb image
        """
        self.num_frames = 16
        self.if_augment = if_augment
        self.is_val = is_val
        self.data_root = root
        self.multi_crop_test = multi_crop_test
        self.dataset_object = WebmDataset(file_input, file_labels, root, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.model = model
        self.num_boxes = num_boxes
        self.size_h = crop[0]
        self.size_w = crop[1]
        # Prepare data for the data loader
        self.prepare_data()
        self.args = args
        self.pre_resize_shape = (256, 340)

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.save_path = save_path
        # Transformations
        if not self.is_val:
            self.transforms = [
                gtransforms.GroupResize((224, 224)),
            ]
        elif self.multi_crop_test:
            self.transforms = [
                gtransforms.GroupResize((256, 256)),
                gtransforms.GroupRandomCrop((256, 256)),
            ]
        else:
            self.transforms = [
                gtransforms.GroupResize((224, 224))
            ]
        self.transforms += [
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(self.img_mean, self.img_std),
        ]
        self.transforms = Compose(self.transforms)

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1],
                                                                   max_distort=0,
                                                                   center_crop_only=True)
        else:
            self.random_crop = None

        boxes_path = args.tracked_boxes
        print('... Loading box annotations might take a minute ...')
        with open(boxes_path, 'r') as f:
            self.box_annotations = json.load(f)

    def prepare_data(self):
        """
        This function creates 2 lists: vid_names, labels
        :return:
        """
        print('Preparing data...')
        vid_names = []
        labels = []
        for i, listdata in enumerate(self.json_data):

            try:
                vid_names.append(listdata.id)
                labels.append(listdata.label)

            except Exception as e:
                print(str(e))

        self.vid_names = vid_names
        self.labels = labels

    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:  the index of video
        """

        folder_id = str(int(self.vid_names[index]))
        # print(folder_id)
        video_data = self.box_annotations[folder_id]     # self.box_annotations['151201']

        list_txt = self.save_path + 'list/' + folder_id + '.txt'
        coord_frame_list = np.loadtxt(list_txt)
        videodata_path = self.save_path + 'frames/' + folder_id + '/'

        # union the objects of two frames
        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_data[int(frame_id)]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                object_set.add(standard_category)
        object_set = sorted(list(object_set))

        if 'hand' in object_set:
            hand_id = torch.tensor(object_set.index('hand'))
        else:
            hand_id = torch.tensor(5)

        frames = []
        for fidx in coord_frame_list:
            image_path = videodata_path + '%d' % fidx + '.jpg'
            frame_sample = Image.open(image_path).convert('RGB')
            frames.append(frame_sample)
            # break  # only one image
        height, width = frames[0].height, frames[0].width

        frames_resize = [img.resize((self.pre_resize_shape[1], self.pre_resize_shape[0]), Image.BILINEAR) for img in frames]

        if self.random_crop is not None:
            # frames, (offset_h, offset_w, crop_h, crop_w) = self.random_crop(frames)
            _, (offset_h, offset_w, crop_h, crop_w) = self.random_crop(frames_resize)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        box_tensors = torch.zeros((self.num_frames, self.num_boxes, 4), dtype=torch.float32) # (cx, cy, w, h)
        box_categories = torch.zeros((self.num_frames, self.num_boxes))

        box_rgb = torch.zeros((self.num_frames*2, self.num_boxes, 3, self.size_h, self.size_w))
        for frame_index, frame_id in enumerate(coord_frame_list):
            frame_id = int(frame_id)
            frame_rgb = frames[frame_index]
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}

            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                global_box_id = object_set.index(standard_category)

                box_coord = box_data['box2d']       # coord value
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord['x2'], box_coord['y2']

                # rgb_box_crop
                crop_frame = frame_rgb.crop((x0, y0, x1, y1))
                crop_img = crop_frame.resize((self.size_h, self.size_w), Image.BILINEAR)  # [H,W]
                loader = transforms.ToTensor()
                crop_img = loader(crop_img)  # [3, H, W] tensor

                box_rgb[2 * frame_index, global_box_id] = crop_img.float()   # [T, V, D, H, W]
                box_rgb[2 * frame_index+1, global_box_id] = crop_img.float()

                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w-1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h-1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

                # (cx, cy, w, h)
                gt_box = np.array([(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box /= 224

                box_tensors[frame_index, global_box_id] = torch.tensor(gt_box).float()

                box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] == 'hand' else 2

                x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))

        return box_rgb, box_tensors, box_categories, hand_id

    def __getitem__(self, index):

        box_rgb, box_tensors, box_categories, hand_id= self.sample_single(index)

        return box_rgb, box_tensors, box_categories, hand_id, self.labels[index]

    def __len__(self):
        return len(self.json_data)
        # return 10000



