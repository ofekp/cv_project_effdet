import time
from multiprocessing import Lock, Value
import helpers
import torch
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
import common
import numpy as np


class BusDataset(BaseDataset):
    def __init__(self, main_folder_path, data_df, num_classes, target_dim, model_name, is_colab, transforms=None,
                 gather_statistics=True):
        self.main_folder_path = main_folder_path
        self.data_df = data_df
        self.num_classes = num_classes
        self.target_dim = target_dim
        self.is_colab = is_colab
        self.transforms = transforms
        self.model_name = model_name
        self.image_ids = data_df['image_id'].unique()
        self.skipped_images = []
        self.gather_statistics = gather_statistics
        if self.gather_statistics:
            self.lock = Lock()
            self.images_processed = Value('i', 0)
            self.total_transform_time = Value('f', 0.0)
            self.total_mask_time = Value('f', 0.0)
            self.total_box_time = Value('f', 0.0)
            self.total_process_time = Value('f', 0.0)
            self.total_image_load_time = Value('f', 0.0)

    def inc_by(self, lock, counter, val):
        lock.acquire()
        try:
            counter.value += val
        finally:
            lock.release()

    def show_stats(self):
        images_processed = self.images_processed.value
        total_process_time = self.total_process_time.value
        avg_time_per_image = 0 if images_processed == 0 else total_process_time / images_processed
        avg_image_load_time = 0 if images_processed == 0 else self.total_image_load_time.value / images_processed
        avg_transform_time = 0 if images_processed == 0 else self.total_transform_time.value / images_processed
        avg_mask_time = 0 if images_processed == 0 else self.total_mask_time.value / images_processed
        avg_box_time = 0 if images_processed == 0 else self.total_box_time.value / images_processed
        print("Processed [{}] images in [{}] seconds"
              " Avg per image [{}]"
              " Avg image load time [{}]"
              " Avg transform time [{}]"
              " Avg mask time [{}]"
              " Avg box time [{}]"
            .format(
            images_processed,
            total_process_time,
            avg_time_per_image,
            avg_image_load_time,
            avg_transform_time,
            avg_mask_time,
            avg_box_time))

    def __getitem__(self, idx):
        if self.gather_statistics:
            start = time.time()
        image_id = self.image_ids[idx]
        vis_df = self.data_df[self.data_df['image_id'] == image_id]
        vis_df = vis_df.astype('int32')
        vis_df = vis_df.reset_index(drop=True)
        labels = torch.tensor(vis_df['label'].tolist())

        box_start_ts = time.time()
        boxes = torch.from_numpy(vis_df[['xmin', 'ymin', 'width', 'height']].values).int()
        # convert to [xmin, ymin, xmax, ymax] which is what efficient det expects
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # rescale bbox to match the given self.target_dim

        if self.gather_statistics:
            self.inc_by(self.lock, self.total_box_time, time.time() - box_start_ts)

        num_objs = len(labels)

        image_id_idx = idx

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        labels_field = 'cls'  # 'labels'
        if "faster" in self.model_name:
            target[labels_field] = torch.sub(labels, 1)
            assert torch.min(target[labels_field]) >= 0
            assert torch.max(target[labels_field]) <= self.num_classes - 1
            target[labels_field] = torch.sub(labels, 1).int().numpy()
        else:
            # we only need the correction for the modified model
            target[labels_field] = labels  # torch.add(labels, 1)  # refer to fast_collate, this is needed for efficient det
            assert torch.min(target[labels_field]) >= 1
            assert torch.max(target[labels_field]) <= self.num_classes
            target[labels_field] = labels.int().numpy()
        # target["boxes"] = boxes
        target["image_id"] = image_id_idx
        target["iscrowd"] = iscrowd.numpy()
        #         target["image_id"] = torch.tensor(image_id_idx)
        #         target["area"] = torch.tensor(area)
        #         target["iscrowd"] = torch.tensor(iscrowd)

        image_load_start_ts = time.time()
        image_orig = Image.open(common.get_image_path(self.main_folder_path, image_id, self.is_colab)).convert("RGB")
        image, boxes_scaled = helpers.rescale(image_orig, boxes, target_dim=self.target_dim)
        # print("Image after rescale 1: shape [{}] min [{}] max [{}]".format(image.shape, torch.min(image), torch.max(image)))
        # boxes_scaled[:, 2] = boxes_scaled[:, 2] - boxes_scaled[:, 0]
        # boxes_scaled[:, 3] = boxes_scaled[:, 3] - boxes_scaled[:, 1]
        target["bbox"] = torch.round(boxes_scaled).double().numpy()
        area = (target["bbox"][:, 3] - target["bbox"][:, 1]) * (target["bbox"][:, 2] - target["bbox"][:, 0])
        target["bbox"] = target["bbox"][:, [1,0,3,2]]  # YXYX
        target["area"] = area
        if self.gather_statistics:
            self.inc_by(self.lock, self.total_image_load_time, time.time() - image_load_start_ts)

        # TODO(ofekp): check what happens here when the image is < self.target_dim. What will helpers.py scale method do to the image in this case?
        target["img_size"] = image_orig.size[-2:] if self.target_dim is None else (self.target_dim, self.target_dim)
        image_orig_max_dim = max(target["img_size"])
        img_scale = self.target_dim / image_orig_max_dim
        target["img_scale"] = 1. / img_scale  # back to original size

        if self.gather_statistics:
            transform_start_ts = time.time()
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.gather_statistics:
            self.inc_by(self.lock, self.total_transform_time, time.time() - transform_start_ts)
            self.inc_by(self.lock, self.images_processed, 1)
            self.inc_by(self.lock, self.total_process_time, time.time() - start)

        assert image.shape[0] <= self.target_dim and image.shape[1] <= self.target_dim and image.shape[2] <= self.target_dim
        image = image * 255
        image = image.numpy().astype(np.uint8)  # CHW
        # print("Image after rescale 2: shape [{}] min [{}] max [{}]".format(image.shape, np.min(image), np.max(image)))
        return image, target

    def __len__(self):
        return len(self.image_ids)
