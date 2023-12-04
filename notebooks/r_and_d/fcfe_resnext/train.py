import pandas as pd

from torch import nn
from notebooks.r_and_d.net_trainer import *
from forest_cover_change_detection.models.fc_fe_resnext import FCFEResNeXt

if __name__ == "__main__":
    config = Config('../../../data/annotated/',
                    '../../../data/train.csv',
                    '../../../data/annotated/test.csv',
                    FCFEResNeXt(6, 2),
                    nn.NLLLoss,
                    100,
                    32,
                    restore_best=True,
                    concat=True)
    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
