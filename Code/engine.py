import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils

import visualize


def train_one_epoch(model, optimizer, data_loader, device, epoch, gradient_accumulation_steps, print_freq,
                    box_threshold):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    optimizer.zero_grad()  # gradient_accumulation
    steps = 0  # gradient_accumulation
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # print("target: {}".format(targets))

        steps += 1  # gradient_accumulation
        # images = list(image.to(device) for image in images)
        # images = torch.stack(images).to(device)
        # images = images.to(device)
        # targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        # targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}

        # vis = visualize.Visualize('.', targets['img_size'][0][0])
        # num_of_detections = len(torch.where(targets['cls'][0] > -1)[0])
        # vis.show_image_data(images[0], targets['cls'][0,:num_of_detections].int(), None, targets['bbox'][0,:num_of_detections,[1,0,3,2]])

        if box_threshold is None:
            loss_dict = model(images, targets)
        else:
            # loss_dict = model(images, box_threshold, targets)
            loss_dict = model(images, targets)

        # losses = sum(loss / gradient_accumulation_steps for loss in loss_dict.values())  # gradient_accumulation
        losses = loss_dict['loss'] / gradient_accumulation_steps

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # optimizer.zero_grad()
        losses.backward()

        # ofekp: we add grad clipping here to avoid instabilities in training
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        # gradient_accumulation
        if steps % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, box_threshold=0.001):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset, box_threshold)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        # images = list(img.to(device) for img in images)
        # targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        # if box_threshold is None:
        #     outputs = model(images)
        # else:
        #     outputs = model(images, box_threshold)

        outputs = model(images, targets)
        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        predictions = outputs['detections'].to(cpu_device)
        batch_predictions = []
        batch_size = predictions.shape[0]
        for i in range(batch_size):
            num_of_detections = len(torch.where(predictions[i][:,4] > 0.0001)[0])
            batch_predictions.insert(i, {
                'boxes': predictions[i][:num_of_detections, 0:4],
                'scores': predictions[i][:num_of_detections, 4],
                'labels': predictions[i][:num_of_detections, 5],
            })
            if num_of_detections > 0:
                try:
                    print("max score was [{}]".format(batch_predictions[0]['scores'][0]))
                except:
                    print("exception when using batch_predictions during eval")
                    print("batch_size [{}]".format(batch_size))
                    print(batch_predictions)

        model_time = time.time() - model_time

        # vis = visualize.Visualize('.', targets['img_size'][0][0])
        # num_of_detections = len(torch.where(targets['cls'][0] > -1)[0])
        # vis.show_image_data(images[0], targets['cls'][0,:num_of_detections].int(), None, targets['bbox'][0,:num_of_detections,[1,0,3,2]])

        # print("img ids: [{}]".format(targets['image_id'].to(cpu_device).tolist()))
        res = {image_id: output for image_id, output in zip(targets['image_id'].to(cpu_device).tolist(), batch_predictions)}  # ofekp: this used to be target["image_id"].item()
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator