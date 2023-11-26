import pandas as pd

from torch import nn
from notebooks.daudt.net_trainer import *
from forest_cover_change_detection.models.fc_ef_res_diff import FCFEResDiff

if __name__ == "__main__":
    config = Config('../../../data/annotated/',
                    '../../../data/train.csv',
                    '../../../data/annotated/test.csv',
                    FCFEResDiff(3, 2),
                    nn.NLLLoss,
                    100,
                    32,
                    multi_in=True,
                    concat=False,
                    restore_best=True)
    df = pd.read_csv(config.test)

    # do(config)
    evaluate(df, config)
