import numpy as np
import ast
import os

def run(myAnnFileName, buses):
    print ('Running dummy runMe')
    annFileNameGT = os.path.join(os.getcwd(),'annotationsTrain.txt')
    writtenAnnsLines = {}
    annFileEstimations = open(myAnnFileName, 'w+')

    annFileGT = open(annFileNameGT, 'r')
    writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())

    for k, line_ in enumerate(writtenAnnsLines['Ground_Truth']):

        line = line_.replace(' ','')
        imName = line.split(':')[0]
        anns_ = line[line.index(':') + 1:].replace('\n', '')
        anns = ast.literal_eval(anns_)
        if (not isinstance(anns, tuple)):
            anns = [anns]
        corruptAnn = [np.round(np.array(x) + np.random.randint(low = 0, high = 100, size = 5)) for x in anns]
        corruptAnn = [x[:4].tolist() + [anns[i][4]] for i,x in enumerate(corruptAnn)]
        strToWrite = imName + ':'
        if(3 <= k <= 5):
            strToWrite += '\n'
        else:
            for i, ann in enumerate(corruptAnn):
                posStr = [str(x) for x in ann]
                posStr = ','.join(posStr)
                strToWrite += '[' + posStr + ']'
                if (i == int(len(anns)) - 1):
                    strToWrite += '\n'
                else:
                    strToWrite += ','
        annFileEstimations.write(strToWrite)


def run2(data_loader_test):
    import train
    import visualize
    import argparse
    import yaml
    import json
    import time
    import torch

    args = None
    with open("Args/args_text.yml", 'r') as args_file:
        args_text = args_file.read()
        parser = argparse.ArgumentParser()
        cfg = yaml.safe_load(args_text)
        parser.set_defaults(**cfg)
        args = parser.parse_args([])

    device = 'cuda:0'
    # device = 'cpu'
    main_folder_path = "."

    # # effdet d0
    # pip uninstall torchvision; pip install git+https://github.com/ofekp/vision.git
    # args.model_name = "tf_efficientdet_d0"
    # args.model_file_suffix = "effdet_h5py_rpn"
    # args.model_file_prefix = "effdet/"
    # args.box_threshold = 0.3

    # effdet d1
    args.model_name = "tf_efficientdet_d2"
    args.model_file_suffix = ""
    args.model_file_prefix = "Model/45AP_d2/"
    args.target_dim = 768
    args.box_threshold = 0.4

    # # effdet d1
    # args.model_name = "tf_efficientdet_d2"
    # args.model_file_suffix = "effdet_d2"
    # args.model_file_prefix = "effdet_d2/"
    # args.box_threshold = 0.30

    # faster
    # pip uninstall torchvision; pip install torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    # args.model_name = "faster"
    # args.model_file_suffix = "faster_sgd"
    # args.model_file_prefix = "faster/"
    # args.box_threshold = None

    # args.box_threshold = None
    # args.model_file_suffix = "faster"
    # args.box_threshold = 0.3
    # args.model_file_suffix = "effdet_checkup"
    # args.data_limit = 2000

    num_classes = 6  # without background class
    train_df, test_df = train.process_data(main_folder_path, num_classes, args.test_size_pct)
    model = train.get_model_detection_efficientdet(args.model_name, num_classes, args.target_dim, freeze_batch_norm=args.freeze_batch_norm_weights)

    train_config = train.TrainConfig(args)
    is_colab = False
    trainer = train.Trainer(main_folder_path, model, train_df, test_df, num_classes, args.target_dim, device, is_colab, config=train_config, args_text=args_text)
    trainer.load_model(device)

    dataset_test = bus_dataset.BusDataset(main_folder_path, test_df, num_classes,
                                          target_dim, config.model_name, False,
                                          T.get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers,
        collate_fn=DetectionFastCollate())
    data_loader_test = PrefetchLoader(data_loader_test)

    cpu_device = torch.device("cpu")
    vis = visualize.Visualize('.', args.target_dim)
    for i, (images, targets) in enumerate(data_loader_test):
        trainer.model.eval()
        with torch.no_grad():
            outputs = model(images, None)
            for output in outputs:
                predictions = output['detections'].to(cpu_device)

    num_of_detections = len(torch.where(targets['cls'][0] > -1)[0])
    vis.show_image_data(images[0], targets['cls'][0,:num_of_detections].int(), None, targets['bbox'][0,:num_of_detections,[1,0,3,2]])

    visualize = visualize.Visualize(main_folder_path, categories_df, args.target_dim)
    img_idx = 1
    trainer.model.eval()
    visualize.show_prediction_on_img(trainer.model, trainer.dataset_test, test_df, img_idx, train.is_colab,
                                     show_groud_truth=False, box_threshold=args.box_threshold, split_segments=True)



    dataset_test = imat_dataset.IMATDataset(main_folder_path, test_df, num_classes, args.target_dim, args.model_name,
                                            False, T.get_transform(train=False))

    for img_idx in range(100, 150):
        #     visualize.show_prediction_on_img(trainer.model, trainer.dataset, train_df, img_idx, train.is_colab, show_groud_truth=True, box_threshold=args.box_threshold, split_segments=True)
        visualize.show_prediction_on_img(trainer.model, dataset_test, test_df, img_idx, train.is_colab,
                                         show_groud_truth=True, box_threshold=0.3, split_segments=True)


# run("newAnns.txt", None)
run2()