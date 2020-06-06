from __future__ import division
from __future__ import print_function

import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from modules.embeddingGenerators.abd.torchreid import models
from modules.embeddingGenerators.alignreid.util.utils import img_to_tensor
from modules.embeddingGenerators.bodyEmbeddingGenerator import BodyEmbeddingGenerator

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'CRITICAL'))

os.environ['TORCH_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.torch'))


class AbdEmbeddingGenerator(BodyEmbeddingGenerator):
    checkpointPath = "../pretrained_models/abd/market_checkpoint_best.pth.tar"

    def get_criterion(num_classes: int, use_gpu: bool, args):
        if args.criterion == 'htri':
            from modules.abd.torchreid.losses.hard_mine_triplet_loss import TripletLoss
            criterion = TripletLoss(num_classes, vars(args), use_gpu)
        elif args.criterion == 'xent':
            from modules.abd.torchreid.losses.cross_entropy_loss import CrossEntropyLoss
            criterion = CrossEntropyLoss(num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
        else:
            raise RuntimeError('Unknown criterion {}'.format(args.criterion))

        return criterion

    def __init__(self):
        self.use_gpu = False
        self.model = models.init_model(name='resnet50', num_classes=751, loss={'xent'}, use_gpu=self.use_gpu)

        try:
            checkpoint = torch.load(self.checkpointPath)
        except Exception as e:
            print(e)
            checkpoint = torch.load(self.checkpointPath, map_location={'cuda:0': 'cpu'})

        pretrain_dict = checkpoint['state_dict']
        model_dict = self.model.state_dict()
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)

        if self.use_gpu:
            self.model = nn.DataParallel(self.model).cuda()

        self.img_transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def getEmbedding(self, bodyImage):
        self.model.eval()
        image = img_to_tensor(Image.fromarray(bodyImage).convert('RGB'), self.img_transform)
        with torch.no_grad():
            if self.use_gpu:
                image = image.cuda()
            features = self.model(image)[0]
            features = features.data.cpu().numpy().flatten()
            return features
