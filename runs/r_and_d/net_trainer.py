import pandas as pd

from dataclasses import dataclass
from tqdm.notebook import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.utils.data import random_split, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from forest_cover_change_detection.dataloaders.change import ChangeDetectionDataset
from forest_cover_change_detection.trainer.train import Compile
from forest_cover_change_detection.metrics.accuracy import *


@dataclass
class Config:

    def __init__(self,
                 root,
                 anno,
                 test,
                 model,
                 loss,
                 epochs,
                 batch_size,
                 concat,
                 multi_in=False,
                 patched=True,
                 restore_best=True,
                 multi_out=False):
        # dataloaders
        self.data_root = root
        self.annotations = anno
        self.test = test
        self.concat = concat
        self.patched = patched

        # models
        self.in_channels = 6
        self.kernel = 3
        self.classes = 2
        self.lr = 0.0001
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=0.001, epochs=epochs, steps_per_epoch=49) # only for tuning
        # self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.15)
        self.multi_in = multi_in
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.restore_best = restore_best
        self.multi_out = multi_out


input_size = 384


def do(config: Config):
    # create dataset
    data_set = ChangeDetectionDataset(config.data_root,
                                      config.annotations,
                                      concat=config.concat,
                                      patched=config.patched
                                      )
    w = torch.load(f'{config.data_root}class_weight.pt')

    print(f"train image count: {len(data_set)}")

    # dataloaders
    train_size = int(len(data_set) * 0.8)
    test_size = len(data_set) - train_size
    train_dataset, test_dataset = random_split(data_set, (train_size, test_size))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size, shuffle=True)

    print(f"training steps: {len(train_dataloader)}  "
          f"validation steps: {len(test_dataloader)}")

    # model creation
    compiled = Compile(config.model,
                       config.optimizer,
                       lr_scheduler=config.scheduler,
                       )

    results = compiled.train(train_dataloader,
                             config.loss(w),
                             config.epochs,
                             test_dataloader,
                             multi_in=config.multi_in,
                             multi_out=config.multi_out)

    # save results
    results.to_csv('./results.csv', index=False)


def get_img_trio(path1, path2, label_path):
    img, gt = (read_image(path1), read_image(path2)), read_image(label_path)

    img1 = resize(img[0], size=[input_size, input_size]) / 255.0
    img2 = resize(img[1], size=[input_size, input_size]) / 255.0
    gt = resize(gt, size=[input_size, input_size]).squeeze(0) / 255.0

    return img1, img2, gt


def evaluate(df, config):
    acc_test = []
    change_acc = []
    no_change_acc = []
    p = []
    r = []
    d = []
    k = []
    metrics = pd.DataFrame()

    # restore best checkpoint
    if config.restore_best:
        state = torch.load(f"./best_model.pth")
        config.model.load_state_dict(state['model_state_dict'])
        config.model = config.model.cuda()

    else:
        state = torch.load("./last-checkpoint.pth")
        config.model.load_state_dict(state['model_state_dict'])
        config.model = config.model.cuda()

    for i, row in tqdm(df.iterrows()):
        img1, img2, gt = get_img_trio(f'{config.data_root}{row.dir}/{row.img_1}',
                                      f'{config.data_root}{row.dir}/{row.img_2}',
                                      f'{config.data_root}{row.label}')

        with torch.no_grad():
            config.model.eval()

            if not config.multi_in:
                if config.multi_out:
                    logits = config.model(torch.cat((img1, img2), dim=0).unsqueeze(0).to('cuda'))[0.55][0].cpu()
                else:
                    logits = config.model(torch.cat((img1, img2), dim=0).unsqueeze(0).to('cuda'))[0].cpu()

            else:
                if config.multi_out:
                    logits = config.model(img1.unsqueeze(0).to('cuda'), img2.unsqueeze(0).to('cuda'))[0.55][0].cpu()
                else:
                    logits = config.model(img1.unsqueeze(0).to('cuda'), img2.unsqueeze(0).to('cuda'))[0].cpu()

            pred = torch.argmax(torch.sigmoid(logits), dim=0)

        # class_acc = class_accuracy(gt, logits)

        tp_, tn_, fp_, fn_ = calculate_confusion(gt, pred)
        acc_test.append(pixel_accuracy(tp_, tn_, fp_, fn_).numpy().tolist())
        p.append(precision(tp_, tn_, fp_, fn_).numpy().tolist())
        r.append(recall(tp_, tn_, fp_, fn_).numpy().tolist())
        d.append(dice(tp_, tn_, fp_, fn_).numpy().tolist())
        k.append(kappa(tp_, tn_, fp_, fn_).numpy().tolist())
        # no_change_acc.append(class_acc[0])
        # change_acc.append(class_acc[1])

    metrics['overall accuracy'] = acc_test
    # metrics['change accuracy'] = change_acc
    # metrics['no change accuracy'] = no_change_acc
    metrics['precision'] = p
    metrics['recall'] = r
    metrics['dice'] = d
    metrics['kappa'] = k

    metrics.to_csv('./metric_eval_v1.csv', index=False)


def evaluate2(df, config):
    acc_test = []
    change_acc = []
    no_change_acc = []
    p = []
    r = []
    d = []
    k = []
    metrics = pd.DataFrame()
    patches_128 = [(0, 128, 0, 128),
                   (128, 256, 0, 128),
                   (256, 384, 0, 128),
                   (0, 128, 128, 256),
                   (128, 256, 128, 256),
                   (256, 384, 128, 256),
                   (0, 128, 256, 384),
                   (128, 256, 256, 384),
                   (256, 384, 256, 384)]
    patches_256 = [(0, 256, 0, 256),
                   (256, 512, 0, 256),
                   (0, 256, 256, 512),
                   (256, 512, 256, 512),
                   ]

    # restore best checkpoint
    if config.restore_best:
        state = torch.load(f"./best_model.pth")
        config.model.load_state_dict(state['model_state_dict'])
        config.model = config.model.cuda()

    else:
        state = torch.load("./last-checkpoint.pth")
        config.model.load_state_dict(state['model_state_dict'])
        config.model = config.model.cuda()

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        tp, tn, fp, fn = 0, 0, 0, 0
        img1, img2, gt = get_img_trio(f'{config.data_root}{row.dir}/{row.img_1}',
                                      f'{config.data_root}{row.dir}/{row.img_2}',
                                      f'{config.data_root}{row.label}')

        for s in patches_128:
            img1_p = img1[:, s[0]:s[1], s[2]:s[3]]
            img2_p = img2[:, s[0]:s[1], s[2]:s[3]]
            gt_p = gt[s[0]:s[1], s[2]:s[3]]

            img1_p = resize(img1_p, size=[384, 384])
            img2_p = resize(img2_p, size=[384, 384])
            gt_p = resize(gt_p.unsqueeze(0), size=[384, 384]).squeeze(0)

            with torch.no_grad():
                config.model.eval()

                if not config.multi_in:
                    if config.multi_out:
                        logits = config.model(torch.cat((img1_p, img2_p), dim=0).unsqueeze(0).to('cuda'))[0.55][0].cpu()
                    else:
                        logits = config.model(torch.cat((img1_p, img2_p), dim=0).unsqueeze(0).to('cuda'))[0].cpu()

                else:
                    if config.multi_out:
                        logits = config.model(img1_p.unsqueeze(0).to('cuda'), img2_p.unsqueeze(0).to('cuda'))[0.55][
                            0].cpu()
                    else:
                        logits = config.model(img1_p.unsqueeze(0).to('cuda'), img2_p.unsqueeze(0).to('cuda'))[0].cpu()

                pred = torch.argmax(torch.sigmoid(logits), dim=0)

            tp_, tn_, fp_, fn_ = calculate_confusion(gt_p, pred)

            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_

        # class_acc = class_accuracy(gt, logits)
        acc_test.append(pixel_accuracy(tp, tn, fp, fn).numpy().tolist())
        p.append(precision(tp, tn, fp, fn).numpy().tolist())
        r.append(recall(tp, tn, fp, fn).numpy().tolist())
        d.append(dice(tp, tn, fp, fn).numpy().tolist())
        k.append(kappa(tp, tn, fp, fn).numpy().tolist())
        # no_change_acc.append(class_acc[0])
        # change_acc.append(class_acc[1])

    metrics['overall accuracy'] = acc_test
    # metrics['change accuracy'] = change_acc
    # metrics['no change accuracy'] = no_change_acc
    metrics['precision'] = p
    metrics['recall'] = r
    metrics['dice'] = d
    metrics['kappa'] = k

    metrics.to_csv('./metric_eval_v2.csv', index=False)
