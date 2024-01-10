from torch import nn
from runs.r_and_d.net_trainer import *
from forest_cover_change_detection.models.v2.multi_in.channel_attention.fc_ef_resnst_se import FCFEResSplitAttentionSE

if __name__ == "__main__":
    model = FCFEResSplitAttentionSE(3, 2)

    state = torch.load(f"./best_model.pth")
    model.load_state_dict(state['model_state_dict'])

    config = Config('../../../../../data/annotated/',
                    '../../../../../data/subsets/256/train.csv',
                    '../../../../../data/annotated/test.csv',
                    model,
                    nn.NLLLoss,
                    10,
                    16,
                    restore_best=True,
                    concat=False,
                    multi_in=True,
                    patched=True)

    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)