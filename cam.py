# https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/vit_example.py
import argparse
import cv2
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset import MVTecAT, SDODataset
from cutpaste import CutPaste
from model import ProjectionNet
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from cutpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.covariance import LedoitWolf
from collections import defaultdict
import pandas as pd
import datetime

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

date_format = '%Y-%m-%dT%H%M%S'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--dataset',
        type=str,
        default='MVTEC',
        help='Dataset (MVTEC, SDO)')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./data/mvtec_anomaly_detection/bottle/test/broken_large/000.png',
        # default='data/aia_171_2012-2016_256/train/2012-01-01T000000__171.jpeg',
        help='Input image path')
    parser.add_argument('--model_path', default="models/model-171-2021-12-03_10_02_17.tch",
                        help='paths of the model (default: models/model-bottle-*.tch)')
    parser.add_argument('--head_layer', default=8, type=int,
                        help='number of layers in the projection head (default: 8)')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    print(f"loading model {args.model_path}")
    head_layer = 2
    head_layers = [512]*head_layer+[128]
    device = "cpu"
    weights = torch.load(args.model_path)
    classes = weights["out.weight"].shape[0]
    model = ProjectionNet(
        pretrained=False, head_layers=head_layers, num_classes=classes)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # target_layers = [list(model.head.children())[-3]]
    target_layers = [model.resnet18.layer4[-1]]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    size = 256

    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))

    if args.dataset == "MVTEC":
        defect_type = "bottle"
        test_data_eval = MVTecAT("data/mvtec_anomaly_detection",
                                 defect_type, size, transform=test_transform, mode="test")

    elif args.dataset == "SDO":
        test_data_eval = SDODataset("data/aia_171_2012-2016_256",
                                    171, size, transform=test_transform, mode="test")

    dataloader_test = DataLoader(test_data_eval, batch_size=64,
                                 shuffle=False, num_workers=0)

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda)  # reshape_transform=reshape_transform
    # cam = GradCAM(model=model, target_layers=target_layers,
    #               use_cuda=args.use_cuda)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category, aug_smooth=True, eigen_smooth=True)

    # grayscale_cam = cam(input_tensor=input_tensor,
    #                     target_category=target_category,
    #                     eigen_smooth=args.eigen_smooth,
    #                     aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(
        f'{args.method}_{datetime.datetime.now().strftime(date_format)}_cam.jpg', cam_image)
