import torch
import matplotlib.pyplot as plt

from forest_cover_change_detection.models.fc_ef import FCFE
from forest_cover_change_detection.models.fc_ef_res import FCFERes
from forest_cover_change_detection.models.fc_ef_res_se import FCFEResSE
from forest_cover_change_detection.models.fc_ef_res_diff import FCFEResDiff
from forest_cover_change_detection.models.fc_fe_resnext import FCFEResNeXt
from forest_cover_change_detection.models.fc_ef_res_with_split_attention import FCFEResSplitAttention
from notebooks.daudt.net_trainer import get_img_trio

models = {'fc_ef': FCFE,
          'fc_ef_res': FCFERes,
          'fc_ef_res_se': FCFEResSE,
          # 'fc_ef_diff': FCFEResDiff,
          'fc_fe_resnext': FCFEResNeXt,
          'fc_ef_res_with_split_attention': FCFEResSplitAttention}

paths = {'fc_ef': './fcfe/best_model.pth',
         'fc_ef_res': './fcfe_res/best_model.pth',
         'fc_ef_res_se': './fcfe_res_se/best_model.pth',
         # 'fc_ef_diff': './fcfe_saim_diff/best_model.pth',
         'fc_fe_resnext': './fcfe_resnext/best_model.pth',
         'fc_ef_res_with_split_attention': './fcfe_res_split_attention/best_model.pth'}


def vis_models_prediction(df):
    t0_imgs = []
    t1_imgs = []
    gts = []
    preds = []

    img_1, img_2, label, dir = df.sample(1).values.tolist()[0]
    img1, img2, gt = get_img_trio(f'../../data/annotated/{dir}/{img_1}',
                                  f'../../data/annotated/{dir}/{img_2}',
                                  f'../../data/annotated/{label}')

    for name, obj in models.items():
        model = obj(6, 2)
        state = torch.load(paths[name])
        model.load_state_dict(state['model_state_dict'])
        model = model.cuda()

        with torch.no_grad():
            model.eval()
            img = torch.cat((img1, img2))
            logits = model(img.unsqueeze(0).to('cuda'))[0].cpu()
            pred = torch.argmax(torch.sigmoid(logits), dim=0)

        t0_imgs.append(img1)
        t1_imgs.append(img2)
        gts.append(gt)
        preds.append(pred)

    fig, axes = plt.subplots(nrows=len(models), ncols=4, figsize=(18, 20))

    for i in range(len(models)):
        axes[i, 0].imshow(t0_imgs[i].permute(1, 2, 0), cmap='gray')
        axes[i, 1].imshow(gts[i], cmap='gray')
        axes[i, 2].imshow(t1_imgs[i].permute(1, 2, 0), cmap='gray')
        axes[i, 3].imshow(preds[i], cmap='gray')
