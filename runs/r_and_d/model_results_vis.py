import pandas as pd
import torch
import matplotlib.pyplot as plt
import tqdm

from forest_cover_change_detection.models.v2.single_in.fc_ef_res_nst import FCFEResSplitAttention as single_resnest
from forest_cover_change_detection.models.v2.multi_in.fc_ef_resnst import FCFEResSplitAttention as multi_resnest
from runs.r_and_d.net_trainer import get_img_trio

models = {'fcef_resnest': single_resnest,
          'siam_resnest': multi_resnest
          }

paths = {'fcef_resnest': './v2/fcef_res_nst/best_model.pth',
         'siam_resnest': './v2/multi_in/fcef_resnst/best_model.pth'
         }


def vis_models_prediction(df, figsize, path=None):
    t0_imgs = []
    t1_imgs = []
    gts = []
    preds = []

    img_1, img_2, label, dir = df.sample(1).values.tolist()[0]
    img1, img2, gt = get_img_trio(f'../../data/annotated/{dir}/{img_1}',
                                  f'../../data/annotated/{dir}/{img_2}',
                                  f'../../data/annotated/{label}')

    for name, obj in tqdm.tqdm(models.items(), desc='predicting'):
        prefix = name.split('_')[0]

        if prefix == 'fcef':
            model = obj(6, 2)

        else:
            model = obj(3, 2)

        state = torch.load(paths[name])
        model.load_state_dict(state['model_state_dict'])
        model = model.cuda()

        with torch.no_grad():
            model.eval()

            if prefix == 'fcef':
                img = torch.cat((img1, img2))
                logits = model(img.unsqueeze(0).to('cuda'))[0].cpu()
                pred = torch.argmax(torch.sigmoid(logits), dim=0)

            else:
                logits = model(img1.unsqueeze(0).to('cuda'), img2.unsqueeze(0).to('cuda'))[0].cpu()
                pred = torch.argmax(torch.sigmoid(logits), dim=0)

        t0_imgs.append(img1)
        t1_imgs.append(img2)
        gts.append(gt)
        preds.append(pred)

    fig, axes = plt.subplots(nrows=len(models), ncols=4, figsize=figsize)

    for i in tqdm.tqdm(range(len(models)), desc='plotting'):
        axes[i, 0].imshow(t0_imgs[i].permute(1, 2, 0), cmap='gray')
        axes[i, 0].axis(False)
        axes[i, 0].set_xlabel('t0_img')

        axes[i, 1].imshow(gts[i], cmap='gray')
        axes[i, 1].axis(False)
        axes[i, 1].set_xlabel('gt')

        axes[i, 2].imshow(t1_imgs[i].permute(1, 2, 0), cmap='gray')
        axes[i, 2].axis(False)
        axes[i, 2].set_xlabel('t1_img')

        axes[i, 3].imshow(preds[i], cmap='gray')
        axes[i, 3].axis(False)
        axes[i, 3].set_xlabel('pred')


    plt.savefig(f'{path}/comp.png')


if __name__ == '__main__':
    test_df = pd.read_csv('../../data/annotated/test.csv')
    vis_models_prediction(test_df, (24, 12), './predictions/experiment_01')
