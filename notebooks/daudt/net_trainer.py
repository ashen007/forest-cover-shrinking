import pandas as pd

from dataclasses import dataclass
from tqdm.notebook import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from forest_cover_change_detection.dataloaders.change import ChangeDetectionDataset
from forest_cover_change_detection.models.fc_siam import FCSiam
from forest_cover_change_detection.trainer.train import Compile
from forest_cover_change_detection.metrics.accuracy import *


@dataclass
class Config:

    def __init__(self,
                 model,
                 loss,
                 epochs,
                 batch_size,
                 checkpointer):
        # dataloaders
        self.data_root = '../../data/annotated'
        self.annotations = '../../data/train.csv'
        self.test = '../../data/annotated/test.csv'
        self.concat = True
        self.patched = True

        # models
        self.in_channels = 6
        self.kernel = 3
        self.classes = 2
        self.lr = 0.001
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, cooldown=10)
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.restore_best = True
        self.checkpointer = checkpointer


def do(config: Config):
    # create dataset
    data_set = ChangeDetectionDataset(config.data_root,
                                      config.annotations,
                                      config.concat,
                                      config.patched
                                      )
    w = torch.load('../../data/annotated/class_weight.pt')

    print(f"train image count: {len(data_set)}")

    # dataloaders
    train_size = int(len(data_set) * 0.8)
    test_size = len(data_set) - train_size
    train_dataset, test_dataset = random_split(data_set, (train_size, test_size))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size, shuffle=True)

    print(f"training steps: {len(train_dataloader)}"
          f"validation steps: {len(test_dataloader)}")

    # model creation
    compiled = Compile(config.model,
                       config.optimizer,
                       lr_scheduler=config.scheduler,
                       checkpointer=config.checkpointer
                       )

    results = compiled.train(train_dataloader,
                             config.loss(w),
                             config.epochs,
                             test_dataloader)

    # save results
    results.to_csv('./results.csv', index=False)

    # restore best checkpoint
    if config.restore_best:
        state = torch.load("../../checkpoints/fcfe/best v1/best_model.pth")
        config.model.load_state_dict(state['model_state_dict'])
        config.model = config.model.cuda()


def evaluate(model, df):
    acc_test = []
    change_acc = []
    no_change_acc = []
    p = []
    r = []
    d = []
    k = []
    metrics = pd.DataFrame()

    for i, row in tqdm(df.iterrows()):
        img, gt = (read_image(f'../../data/annotated/{row.dir}/{row.img_1}'),
                   read_image(f'../../data/annotated/{row.dir}/{row.img_2}')), read_image(
            f'../../data/annotated/{row.label}')

        img1 = resize(img[0], size=[256, 256]) / 255.0
        img2 = resize(img[1], size=[256, 256]) / 255.0
        gt = resize(gt, size=[256, 256]).squeeze(0) / 255.0

        with torch.no_grad():
            model.eval()
            logits = model(torch.cat((img1, img2), dim=0).unsqueeze(0).to('cuda'))[0].cpu()
            pred = torch.argmax(torch.sigmoid(logits), dim=0)

        class_acc = class_accuracy(gt, logits)

        acc_test.append(pixel_accuracy(gt, pred).numpy().tolist())
        p.append(precision(gt, pred).numpy().tolist())
        r.append(recall(gt, pred).numpy().tolist())
        d.append(dice(gt, pred).numpy().tolist())
        k.append(kappa(gt, pred).numpy().tolist())
        no_change_acc.append(class_acc[0])
        change_acc.append(class_acc[1])

    metrics['overall accuracy'] = acc_test
    metrics['change accuracy'] = change_acc
    metrics['no change accuracy'] = no_change_acc
    metrics['precision'] = p
    metrics['recall'] = r
    metrics['dice'] = d
    metrics['kappa'] = k

    metrics.to_csv('./metric_eval.csv')
