from torch import nn
from runs.r_and_d.net_trainer import *
from forest_cover_change_detection.models.v2.multi_in.channel_attention.fc_ef_resnst_se import FCFEResSplitAttentionSE

if __name__ == "__main__":
    config = Config('../../../../../data/annotated/',
                    '../../../../../data/subsets/256/train.csv',
                    '../../../../../data/annotated/test.csv',
                    FCFEResSplitAttentionSE(3, 2),
                    nn.NLLLoss,
                    50,
                    16,
                    restore_best=True,
                    concat=False,
                    multi_in=True,
                    patched=True)

    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
