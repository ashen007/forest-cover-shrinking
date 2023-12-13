from torch import nn
from runs.r_and_d.net_trainer import *
from forest_cover_change_detection.models.v2.single_in.channel_attention.fc_ef_resnst_se import FCFEResSplitAttentionSE

if __name__ == "__main__":
    config = Config('../../../../../data/annotated/',
                    '../../../../../data/annotated/train.csv',
                    '../../../../../data/annotated/test.csv',
                    FCFEResSplitAttentionSE(6, 2),
                    nn.NLLLoss,
                    300,
                    32,
                    restore_best=True,
                    concat=True,
                    patched=False)

    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
