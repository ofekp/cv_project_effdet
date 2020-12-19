import torch
import numpy as np
from PIL import Image
import random
import math

import torchvision.transforms.functional as TF
from torchvision.transforms import functional as F
from torchvision import transforms as T


# following is needed for reproducibility
# refer to https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(17)
np.random.seed(181)
torch.manual_seed(129)


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


class RandomRotate(object):
    def __init__(self, prob):
        self.prob = prob
        self.deg = [90, 10, 30, 45]

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        deg = random.choice(self.deg) * random.choice([-1,1])
        image_new = F.rotate(image, deg)

        rad = math.radians(-deg)
        rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)],
                                    [np.sin(rad), np.cos(rad)]])
        bbox = target["bbox"]  # YXYX
        bbox = bbox[:,[1, 0, 3, 2]]  # XYXY
        bbox = bbox[:, [0, 1, 2, 3, 2, 1, 0, 3]]  # get all corners of all the boxes
        bbox = bbox.reshape(-1, 2).transpose()
        height, width = image.shape[-2:]
        assert height == width
        bbox -= width / 2
        bbox = np.matmul(rotation_matrix, bbox)
        bbox = bbox.transpose().reshape(-1,4,2).transpose(0,2,1)
        bbox_min = np.min(bbox, axis=2)
        bbox_max = np.max(bbox, axis=2)
        bbox = np.column_stack((bbox_min, bbox_max))
        bbox = bbox[:, [1, 0, 3, 2]]  # YXYX
        bbox += width / 2
        target["bbox"] = bbox

        return image_new, target


class RandomSaltAndPepper():
    def __init__(self, prob):
        self.prob = prob
        self.salt_threshold = 0.05
        self.pepper_threshold = 0.05
        self.upperValue = 0.9
        self.lowerValue = 0.1

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        assert image.shape[1] == image.shape[1]
        image_dim = image.shape[1]
        random_matrix = np.random.rand(image_dim, image_dim)
        image[:, random_matrix < self.salt_threshold] = self.upperValue
        image[:, random_matrix > (1.0 - self.pepper_threshold)] = self.lowerValue
        # elif self.noiseType == "RGB":
        #     random_matrix = np.random.random(img.shape)
        #     img[random_matrix >= (1 - self.treshold)] = self.upperValue
        #     img[random_matrix <= self.treshold] = self.lowerValue
        return image, target


class RandomBlackBoxes():
    def __init__(self, prob):
        self.prob = prob
        self.threshold = 0.99999
        self.sizes = [20,50,70,80,100]

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        assert image.shape[1] == image.shape[1]
        image_dim = image.shape[1]
        random_matrix = np.random.rand(image_dim, image_dim)
        coords = np.argwhere(random_matrix >= self.threshold)
        for coord in coords:
            width = random.choice(self.sizes)
            height = random.choice(self.sizes)
            end_coord = np.array([min(coord[0] + width, image_dim), min(coord[1] + height, image_dim)])
            assert coord[0] < end_coord[0]
            assert coord[1] < end_coord[1]
            image[:, coord[0]:end_coord[0], coord[1]:end_coord[1]] = 0
        return image, target


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


class ImageToPIL:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, tensor_img, annotations: dict):
        image_pil = T.ToPILImage()(tensor_img)
        return image_pil, annotations


def get_transform(train):
    transforms = []
#     transforms.append(T.ToTensor())
    if train:
        # transforms.append(ImageToPIL())

        # transforms.append(RandomBlackBoxes(0.3))
        transforms.append(RandomHorizontalFlip(0.3))
        # transforms.append(RandomRotate(0.3))
        transforms.append(RandomSaltAndPepper(0.3))

        # transforms.append(ImageToTensor())  # helpers.scale already coverts to tensor
        # transforms.append(RandomGreyscale(0.1))
    return Compose(transforms)