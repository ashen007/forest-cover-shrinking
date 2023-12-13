from torch import nn
from runs.r_and_d.net_trainer import *
from forest_cover_change_detection.models.v2.single_in.combine_attention.fc_fe_resnext_add import FCFEResNeXtAdditive

if __name__ == "__main__":
    config = Config('../../../../data/annotated/',
                    '../../../../data/annotated/train.csv',
                    '../../../../data/annotated/test.csv',
                    FCFEResNeXtAdditive(6, 2),
                    nn.NLLLoss,
                    300,
                    32,
                    restore_best=True,
                    concat=True,
                    patched=False)

    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
