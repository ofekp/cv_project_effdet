import os
import sys
import yaml
import json
import argparse
import subprocess
import pandas as pd
from datetime import datetime
import numpy as np

import visualize
import bus_dataset
import transforms as T

import torch
import torchvision
from torchvision.ops import MultiScaleRoIAlign
import effdet
from effdet import EfficientDet, load_pretrained, DetBenchTrain
from effdet.efficientdet import HeadNet

import coco_utils, coco_eval, engine, utils


parser = argparse.ArgumentParser(description='Training Config')

# parsing boolean typed arguments
# refer to https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected but got [{}].'.format(v))


parser.add_argument('--train', type=str2bool, default=True, metavar='BOOL',
                    help='Will start training the model (default=True)')
parser.add_argument('--model-name', type=str, default='tf_efficientdet_d1', metavar='MODEL_NAME',
                    help='The name of the model to use as found in EfficientDet model_config.py file (default=tf_efficientdet_d0)')
parser.add_argument('--lr', type=float, default=0.007, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=0.00005, metavar='WEIGHT_DECAY',
                    help='weight decay (default: 0.00005)')
parser.add_argument('--test-size-pct', type=float, default=0.2, metavar='TEST_SIZE_PCT',
                    help='test set size percentage, set to 0.2 for 20%.'
                         ' Minimum is one image per class which is 0.1 (default: 0.2)')
# note the the images provided for the projct are of size 3648 x 2736
parser.add_argument('--target-dim', type=int, default=640, metavar='DIM',  # d0 512, d1 640, d2 768, d3 896, 3072
                    help='Dimention of the images. It is vital that the image size will be devisiable by 2 at least 6 times (default=512)')
parser.add_argument('--freeze-batch-norm-weights', type=str2bool, default=True, metavar='BOOL',
                    help='Freeze batch normalization weights (default=True)')
parser.add_argument('--add-user-name-to-model-file', type=str2bool, default=True, metavar='BOOL',
                    help='Will add the user name to the model file that is saved during training (default=True)')

parser.add_argument('--num-epochs', type=int, default=150, metavar='NUM_EPOCHS',
                    help='number of epochs (default: 150)')
parser.add_argument('--batch-size', type=int, default=4, metavar='BATCH_SIZE',
                    help='batch size (default: 12)')
parser.add_argument('--num-workers', type=int, default=4, metavar='NUM_WORKERS',
                    help='number of workers for the dataloader (default: 4)')
parser.add_argument('--load-model', type=str2bool, default=False, metavar='BOOL',
                    help='Will load a model file (default=False)')
parser.add_argument('--box-threshold', type=float, default=0.3, metavar='BOX_THRESHOLD',
                    help='score threshold - boxes with scores lower than specified will be ignored in training and evaluation (default: 0.3)')
parser.add_argument('--model-file-prefix', type=str, default='', metavar='PREFIX',
                    help='Prefix, may be folder, to load the model file that is saved during training (default=empty string)')

# scheduler params
parser.add_argument('--sched-factor', type=float, default=0.7, metavar='FACTOR',
                    help='scheduler factor (default: 0.7)')
parser.add_argument('--sched-patience', type=int, default=1, metavar='PATIENCE',
                    help='scheduler patience (default: 1)')
parser.add_argument('--sched-verbose', type=str2bool, default=False, metavar='VERBOSE',
                    help='scheduler verbosity (default: False)')
parser.add_argument('--sched-threshold', type=float, default=0.0001, metavar='THRESHOLD',
                    help='scheduler threshold (default: 0.0001)')
parser.add_argument('--sched-min-lr', type=float, default=1e-8, metavar='MIN_LR',
                    help='scheduler min LR (default: 1e-8)')
parser.add_argument('--sched-eps', type=float, default=1e-08, metavar='EPS',
                    help='scheduler epsilon (default: 1e-08)')

# additional params
parser.add_argument('--gradient-accumulation-steps', type=int, default=2, metavar='NUM_EPOCHS',
                    help='number of epoch to accomulate gradients before applying back-prop (default: 2)')  # TODO(ofekp): change to 1?
parser.add_argument('--save-every', type=int, default=5, metavar='NUM_EPOCHS',
                    help='save the model every few epochs (default: 5)')
parser.add_argument('--eval-every', type=int, default=10, metavar='NUM_EPOCHS',
                    help='evaluate and print the evaluation to screen every few epochs (default: 10)')

def parse_args():
    # parse the args that are passed to this script
    args = parser.parse_args()

    # save the args as a text string so we can log them later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def run_os(cmd_as_list):
    process = subprocess.Popen(cmd_as_list,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.strip().decode('utf-8') if stdout is not None else stdout
    stderr = stderr.strip().decode('utf-8') if stderr is not None else stderr
    return stdout, stderr


def print_bold(str):
    print("\033[1m" + str + "\033[0m")


def print_nvidia_smi(device):
    if device == 'cuda:0':
        stderr, _ = run_os(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv'])
        print(stderr)


def process_data(main_folder_path, num_classes, test_set_pct):
    annotations_df = pd.DataFrame(columns=['image_id', 'xmin', 'ymin', 'width', 'height', 'label'])
    with open(main_folder_path + '/annotationsTrain.txt', 'r') as file:
        for line in file:
            split_line = line.split(':')
            image_id = split_line[0].strip().split('.')[0][4:]
            image_annotations = json.loads('{"data": [' + split_line[1].strip() + ']}')
            for annotation in image_annotations['data']:
                annotations_df = annotations_df.append({'image_id': int(image_id), 'xmin': int(annotation[0]), 'ymin': int(annotation[1]), 'width': int(annotation[2]), 'height': int(annotation[3]), 'label': int(annotation[4])}, ignore_index=True)

    print(annotations_df)

    # split the date to train set and validation set
    image_ids = annotations_df['image_id'].unique()
    image_train_count = int(len(image_ids) * (1 - test_set_pct))
    image_ids_train = image_ids[:image_train_count]
    image_ids_test = image_ids[image_train_count:]
    assert len(image_ids) == (len(image_ids_test) + len(image_ids_train))
    train_df = annotations_df[annotations_df['image_id'].isin(image_ids_train)]
    test_df = annotations_df[annotations_df['image_id'].isin(image_ids_test)]
    assert len(annotations_df) == (len(test_df) + len(train_df))
    # the split must also take into consideration that all labels appear in both sets at least once
    assert len(test_df['label'].unique()) == num_classes
    assert len(train_df['label'].unique()) == num_classes

    return train_df, test_df


def set_bn_eval(m):
    classname = m.__class__.__name__
    if "BatchNorm2d" in classname:
        m.affine = False
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m.eval()


def freeze_bn(model):
    model.apply(set_bn_eval)


def get_model_detection_efficientdet(model_name, num_classes, target_dim, freeze_batch_norm=False):
    print("Using EffDet detection model")

    config = effdet.get_efficientdet_config(model_name)
    efficientDetModel = EfficientDet(config, pretrained_backbone=False)
    load_pretrained(efficientDetModel, config.url)
    import omegaconf
    with omegaconf.read_write(config):
        config.num_classes = num_classes
        # config.image_size = target_dim
    efficientDetModel.class_net = HeadNet(config, num_outputs=num_classes)

    if freeze_batch_norm:
        # we only freeze BN layers in backbone and the BiFPN
        print("Freezing batch normalization weights")
        freeze_bn(efficientDetModel.backbone)

    with omegaconf.read_write(efficientDetModel.config):
        efficientDetModel.config.num_classes = num_classes

    # print(DetBenchTrain(efficientDetModel, config))
    return DetBenchTrain(efficientDetModel, config)


class TrainConfig:
    def __init__(self, args):
        if args.add_user_name_to_model_file:
            self.model_file_suffix = os.getlogin() + "_" + args.model_name
        else:
            self.model_file_suffix = args.model_name
        self.model_file_prefix = args.model_file_prefix
        self.verbose = True
        self.save_every = args.save_every
        self.eval_every = args.eval_every
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_epochs = args.num_epochs
        self.model_name = args.model_name
        self.box_threshold = args.box_threshold

        if "faster" in self.model_name:
            # special case of training the conventional model based on Faster R-CNN
            self.optimizer_class = torch.optim.SGD
            self.optimizer_config = dict(
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        else:
            print("Using AdamW optimizer")
            self.optimizer_class = torch.optim.AdamW
            self.optimizer_config = dict(
                lr=args.lr,
                weight_decay=args.weight_decay
            )

        if "_d0" in self.model_name:
            print("Using StepLR")
            self.scheduler_class = torch.optim.lr_scheduler.StepLR
            self.scheduler_config = dict(
                step_size=10,
                gamma=0.2
            )
        else:
            print("Using ReduceLROnPlateau")
            self.scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
            self.scheduler_config = dict(
                mode='min',
                factor=args.sched_factor,
                patience=args.sched_patience,
                verbose=False,
                threshold=args.sched_threshold,
                threshold_mode='abs',
                cooldown=0,
                min_lr=args.sched_min_lr,
                eps=args.sched_eps
            )


def my_fast_collate(targets):
    MAX_NUM_INSTANCES = 20
    batch_size = len(targets)

    # FIXME this needs to be more robust
    target = dict()
    for k, v in targets[0].items():
        if torch.is_tensor(v):
            target_shape = (batch_size, MAX_NUM_INSTANCES)
            if len(v.shape) > 1:
                target_shape = (batch_size, MAX_NUM_INSTANCES) + v.shape[1:]
            target_dtype = v.dtype
        elif isinstance(v, np.ndarray):
            # if a numpy array, assume it relates to object instances, pad to MAX_NUM_INSTANCES
            target_shape = (batch_size, MAX_NUM_INSTANCES)
            if len(v.shape) > 1:
                target_shape = target_shape + v.shape[1:]
            target_dtype = torch.float32
        elif isinstance(v, (tuple, list)):
            # if tuple or list, assume per batch
            target_shape = (batch_size, len(v))
            target_dtype = torch.float32 if isinstance(v[0], float) else torch.int32
        else:
            # scalar, assume per batch
            target_shape = batch_size
            target_dtype = torch.float32 if isinstance(v, float) else torch.int64
        target[k] = torch.zeros(target_shape, dtype=target_dtype)

    for i in range(batch_size):
        for tk, tv in targets[i].items():
            if torch.is_tensor(tv):
                target[tk][i, 0:tv.shape[0]] = tv
            elif isinstance(tv, np.ndarray) and len(tv.shape):
                target[tk][i, 0:tv.shape[0]] = torch.from_numpy(tv)
            else:
                target[tk][i] = torch.tensor(tv, dtype=target[tk].dtype)

    return target


class DetectionFastCollate:
    """ A detection specific, optimized collate function w/ a bit of state.
    Optionally performs anchor labelling. Doing this here offloads some work from the
    GPU and the main training process thread and increases the load on the dataloader
    threads.
    """
    def __init__(
            self,
            instance_keys=None,
            instance_shapes=None,
            instance_fill=-1,
            max_instances=20,
            anchor_labeler=None,
    ):
        instance_keys = instance_keys or {'bbox', 'bbox_ignore', 'cls', 'area', 'iscrowd'}
        instance_shapes = instance_shapes or dict(
            bbox=(max_instances, 4), bbox_ignore=(max_instances, 4), cls=(max_instances,), area=(max_instances,), iscrowd=(max_instances,))
        self.instance_info = {k: dict(fill=instance_fill, shape=instance_shapes[k]) for k in instance_keys}
        self.max_instances = max_instances
        self.anchor_labeler = anchor_labeler

    def __call__(self, batch):
        batch_size = len(batch)
        target = dict()
        labeler_outputs = dict()
        img_tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            img_tensor[i] += torch.from_numpy(batch[i][0])
            labeler_inputs = {}
            for tk, tv in batch[i][1].items():
                instance_info = self.instance_info.get(tk, None)
                if instance_info is not None:
                    # target tensor is associated with a detection instance
                    tv = torch.from_numpy(tv).to(dtype=torch.float32)
                    if self.anchor_labeler is None:
                        if i == 0:
                            shape = (batch_size,) + instance_info['shape']
                            target_tensor = torch.full(shape, instance_info['fill'], dtype=torch.float32)
                            target[tk] = target_tensor
                        else:
                            target_tensor = target[tk]
                        num_elem = min(tv.shape[0], self.max_instances)
                        target_tensor[i, 0:num_elem] = tv[0:num_elem]
                    else:
                        # no need to pass gt tensors through when labeler in use
                        if tk in ('bbox', 'cls'):
                            labeler_inputs[tk] = tv
                else:
                    # target tensor is an image-level annotation / metadata
                    if i == 0:
                        # first batch elem, create destination tensors
                        if isinstance(tv, (tuple, list)):
                            # per batch elem sequence
                            shape = (batch_size, len(tv))
                            dtype = torch.float32 if isinstance(tv[0], (float, np.floating)) else torch.int32
                        else:
                            # per batch elem scalar
                            shape = batch_size,
                            dtype = torch.float32 if isinstance(tv, (float, np.floating)) else torch.int64
                        target_tensor = torch.zeros(shape, dtype=dtype)
                        target[tk] = target_tensor
                    else:
                        target_tensor = target[tk]
                    target_tensor[i] = torch.tensor(tv, dtype=target_tensor.dtype)

            if self.anchor_labeler is not None:
                cls_targets, box_targets, num_positives = self.anchor_labeler.label_anchors(
                    labeler_inputs['bbox'], labeler_inputs['cls'], filter_valid=False)
                if i == 0:
                    # first batch elem, create destination tensors, separate key per level
                    for j, (ct, bt) in enumerate(zip(cls_targets, box_targets)):
                        labeler_outputs[f'label_cls_{j}'] = torch.zeros(
                            (batch_size,) + ct.shape, dtype=torch.int64)
                        labeler_outputs[f'label_bbox_{j}'] = torch.zeros(
                            (batch_size,) + bt.shape, dtype=torch.float32)
                    labeler_outputs['label_num_positives'] = torch.zeros(batch_size)
                for j, (ct, bt) in enumerate(zip(cls_targets, box_targets)):
                    labeler_outputs[f'label_cls_{j}'][i] = ct
                    labeler_outputs[f'label_bbox_{j}'][i] = bt
                labeler_outputs['label_num_positives'][i] = num_positives
        if labeler_outputs:
            target.update(labeler_outputs)

        return img_tensor, target


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class PrefetchLoader:

    def __init__(self,
            loader,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            re_prob=0.,
            re_mode='pixel',
            re_count=1,
            ):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        if re_prob > 0.:
            self.random_erasing = RandomErasing(probability=re_prob, mode=re_mode, max_count=re_count)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input = next_input.float().sub_(self.mean).div_(self.std)
                next_target = {k: v.cuda(non_blocking=True) for k, v in next_target.items()}
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input, next_target)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class Trainer:

    def __init__(self, main_folder_path, model, train_df, test_df, num_classes, target_dim, device, is_colab, config):
        self.main_folder_path = main_folder_path
        self.model = model
        self.train_df = train_df
        self.test_df = test_df
        self.device = device
        self.config = config
        self.num_classes = num_classes
        self.target_dim = target_dim
        self.is_colab = is_colab
        if "faster" in self.config.model_name:
            # special case of training the conventional model based on Faster R-CNN
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = self.config.optimizer_class(params, **self.config.optimizer_config)
        else:
            self.optimizer = self.config.optimizer_class(self.model.parameters(), **self.config.optimizer_config)
        self.scheduler = self.config.scheduler_class(self.optimizer, **self.config.scheduler_config)
        self.model_file_path = self.get_model_file_path(self.is_colab, prefix=config.model_file_prefix,
                                                        suffix=config.model_file_suffix)
        self.log_file_path = self.get_log_file_path(self.is_colab, suffix=config.model_file_suffix)
        self.epoch = 0
        self.visualize = visualize.Visualize(self.main_folder_path, self.target_dim, dest_folder='Images')

        # use our dataset and defined transformations
        self.dataset = bus_dataset.BusDataset(self.main_folder_path, self.train_df, self.num_classes,
                                              self.target_dim, self.config.model_name, False,
                                              T.get_transform(train=True))
        self.dataset_test = bus_dataset.BusDataset(self.main_folder_path, self.test_df, self.num_classes,
                                                   self.target_dim, self.config.model_name, False,
                                                   T.get_transform(train=False))

        # TODO(ofekp): do we need this?
        # split the dataset in train and test set
        # indices = torch.randperm(len(dataset)).tolist()
        # dataset = torch.utils.data.Subset(dataset, indices[:-50])
        # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        self.log('Trainer initiallized. Device is [{}]'.format(self.device))

    def get_model_identifier(self):
        return 'dim_' + str(self.target_dim) + '_classes_' + str(self.num_classes)

    def get_model_file_path(self, is_colab, prefix=None, suffix=None):
        model_file_path = self.get_model_identifier()
        if prefix:
            model_file_path = prefix + ('' if prefix[-1] == '/' else '_') + model_file_path
        if suffix:
            model_file_path = model_file_path + '_' + suffix

        model_file_path = 'Model/' + model_file_path + '.model'

        if is_colab:
            model_file_path = self.main_folder_path + 'code_ofek/' + model_file_path
        else:
            model_file_path = self.main_folder_path + '/' + model_file_path

        return model_file_path

    def get_log_file_path(self, is_colab, prefix=None, suffix=None):
        log_file_path = self.get_model_identifier()
        if prefix:
            log_file_path = prefix + ('' if prefix[-1] == '/' else '_') + log_file_path
        if suffix:
            log_file_path = log_file_path + '_' + suffix

        log_file_path = 'Log/' + log_file_path + '.log'

        if is_colab:
            log_file_path = self.main_folder_path + 'code_ofek/' + log_file_path
        else:
            log_file_path = self.main_folder_path + '/' + log_file_path

        return log_file_path

    def load_model(self, device):
        if not os.path.isfile(self.model_file_path):
            self.log("Cannot load model file [{}] since it does not exist".format(self.model_file_path))
            return False
        checkpoint = torch.load(self.model_file_path)  # map_location=device
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # model must be moved to device before we init the optimizer otherwise loading a model and training
        # again will procduce the "both cpu and cuda" error, refer to the solution in this thread:
        # https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783
        # I also added the solution to this issue https://github.com/pytorch/pytorch/issues/34470
        self.model.to(device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # TODO(ofekp): uncomment
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # TODO(ofekp): uncomment
        #         self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        self.log("Loaded model file [{}] trained epochs [{}]".format(self.model_file_path, checkpoint['epoch']))
        return True

    def save_model(self):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            #             'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, self.model_file_path)
        self.log('Saved model to [{}]'.format(self.model_file_path))
        print_nvidia_smi(self.device)
        self.dataset_test.show_stats()

    def eval_model(self, data_loader_test):
        self.model.eval()
        with torch.no_grad():
            img_idx = 0
            self.visualize.show_prediction_on_img(self.model, data_loader_test, self.dataset_test, self.test_df, img_idx, self.is_colab,
                                                  show_groud_truth=False, box_threshold=self.config.box_threshold,
                                                  split_segments=False)
            # evaluate on the test dataset
            if "faster" in self.config.model_name:
                # special case of training the conventional model based on Faster R-CNN
                engine.evaluate(self.model, data_loader_test, device=self.device, box_threshold=None)
            else:
                engine.evaluate(self.model, data_loader_test, device=self.device)

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_file_path, 'a+') as logger:
            logger.write(f'{message}\n')

    def train(self):
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers,
            collate_fn = DetectionFastCollate())
            # collate_fn=utils.collate_fn)
        data_loader = PrefetchLoader(data_loader)

        data_loader_test = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers,
            collate_fn = DetectionFastCollate())
            # collate_fn=utils.collate_fn)
        data_loader_test = PrefetchLoader(data_loader_test)

        for _ in range(self.config.num_epochs):
            # train one epoch
            metric_logger = engine.train_one_epoch(
                self.model,
                self.optimizer,
                data_loader,
                self.device,
                self.epoch,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                print_freq=100,
                box_threshold=self.config.box_threshold)

            # update the learning rate
            if "_d0" in self.config.model_name:
                print("Updating StepLR")
                self.scheduler.step()
            else:
                print("Updating ReduceLROnPlateau")
                self.scheduler.step(metric_logger.__getattr__('loss').avg)
            torch.cuda.empty_cache()  # ofekp: attempting to avoid GPU memory usage increase

            if (self.epoch) % self.config.save_every == 0:
                self.save_model()

            if (self.epoch) % self.config.eval_every == 0:
                self.eval_model(data_loader_test)

            self.log("Epoch [{}/{}]".format(self.epoch + 1, self.config.num_epochs))
            self.epoch += 1

        self.log("Saving model one last time")
        self.save_model()
        self.eval_model(data_loader_test)
        self.log("That's it!")


def main():
    args, args_text = parse_args()
    main_folder_path = "."  # assuming running from Code folder

    # create folders if needed
    needed_folders = ["./Model/", "./Log/"]
    for needed_folder in needed_folders:
        if not os.path.exists(needed_folder):
            os.mkdir(needed_folder)

    # prepare a log file
    now = datetime.now() # current date and time
    date_str = now.strftime("%Y%m%d%H%M")
    log_file_path = "./Log/" + date_str + ".log"
    log_file = open(log_file_path, "a")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file

    # device
    forceCPU = False  # TODO: argparse
    if forceCPU:
        device = 'cpu'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Device type [{}]".format(device))
    if device == 'cuda:0':
        print("Device description [{}]".format(torch.cuda.get_device_name(0)))

    num_classes = 6  # without background class
    train_df, test_df = process_data(main_folder_path, num_classes, args.test_size_pct)

    model = get_model_detection_efficientdet(args.model_name, num_classes, args.target_dim, freeze_batch_norm=args.freeze_batch_norm_weights)

    print("got model")

    # get the model using our helper function
    train_config = TrainConfig(args)
    is_colab = False
    trainer = Trainer(main_folder_path, model, train_df, test_df, num_classes, args.target_dim, device, is_colab, config=train_config)

    # load a saved model
    if args.load_model:
        if not trainer.load_model(device):
            exit(1)

    if args.train:
        print_nvidia_smi(device)
        model.to(device)
        print_nvidia_smi(device)
        trainer.train()

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()



if __name__ == '__main__':
    main()
