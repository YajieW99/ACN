import os
import numpy as np
from PIL import Image
import torch
from data_utils.data_parser import WebmDataset
import skvideo.io
from ipdb import set_trace


class Video10Extract(torch.utils.data.Dataset):
    """
    Something-Something dataset based on *frames* extraction
    """

    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 save_path,
                 is_test=False):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param is_test: is_test flag
        """
        self.data_root = root
        self.dataset_object = WebmDataset(file_input, file_labels, root, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        # Prepare data for the data loader
        self.prepare_data()
        self.save_path = save_path

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        :return:
        """
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
        print('Prepare_data Finished')

    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:  the index of video
        :return:
        """

        folder_id = str(int(self.vid_names[index]))

        videoframes = skvideo.io.vread(self.data_root + folder_id + '.mp4')    # [T, H, W, D]
        n_frame = videoframes.shape[0]
        s = np.linspace(0,n_frame-1,num=16)
        # all frames
        coord_frame_list0 = [int(round(p)) for p in range(n_frame)]
        # 16 frames
        coord_frame_list = [int(round(p)) for p in s]

        image_path = self.save_path + 'frames/' + folder_id + '/'
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        for fidx in coord_frame_list0:
            frame_sample = Image.fromarray(np.uint8(videoframes[fidx]))
            frame_sample.save(image_path + '/' + str(fidx) + '.jpg')
        x = np.array(coord_frame_list)
        txt_path = self.save_path + 'list/'
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)
        np.savetxt(self.save_path + 'list/' + folder_id + '.txt', x, fmt='%d')

        return n_frame

    def __getitem__(self, index):

        n_frame = self.sample_single(index)
        return n_frame

    def __len__(self):
        return len(self.json_data)
