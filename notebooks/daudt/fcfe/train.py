import pandas as pd

from torch import nn
from notebooks.daudt.net_trainer import *
from forest_cover_change_detection.models.fc_ef import FCFE
from forest_cover_change_detection.utils.save_best_cp import SaveBestCheckPoint

if __name__ == "__main__":
    config = Config('../../../data/annotated/',
                    '../../../data/train.csv',
                    '../../../data/annotated/test.csv',
                    FCFE(6, 3, 2),
                    nn.NLLLoss,
                    100,
                    32,
                    checkpointer=SaveBestCheckPoint('../../../checkpoints/fcfe'),
                    restore_best=False)
    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
