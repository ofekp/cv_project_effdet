import random
import torch
import numpy as np

from torchvision.transforms import functional as F
from torchvision import transforms


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __len__(self):
        return len(self.transforms)


# getting YXYX
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["bbox"]
            bbox[:, [1, 3]] = width - bbox[:, [3, 1]]
            target["bbox"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


# class RandomGreyscale(object):
#     def __init__(self, prob):
#         self.prob = prob
#
#     def __call__(self, image, target):
#         if random.random() < self.prob:
#             image = transforms.ToPILImage(mode='RGB')(image)
#             greyscale_transform = transforms.Grayscale(num_output_channels=3)
#             image = greyscale_transform(image)
#             trans = transforms.ToTensor()
#             image = trans(image)
#         return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ImageToNumpy:
    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img, annotations


class ImageToTensor:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        # np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype), annotations


def get_transform(train):
    transforms = []
#     transforms.append(T.ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        # transforms.append(ImageToTensor())  # helpers.scale already coverts to tensor
        # transforms.append(RandomGreyscale(0.1))
    return Compose(transforms)