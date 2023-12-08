import pandas as pd

from torch import nn
from notebooks.r_and_d.net_trainer import *
from forest_cover_change_detection.models.fc_siam import FCSiam
from forest_cover_change_detection.utils.save_best_cp import SaveBestCheckPoint

if __name__ == "__main__":
    config = Config('../../../data/annotated/',
                    '../../../data/train.csv',
                    '../../../data/annotated/test.csv',
                    FCSiam(3, 2, False),
                    nn.NLLLoss,
                    100,
                    32,
                    multi_in=True,
                    concat=False,
                    checkpointer=SaveBestCheckPoint('../../../checkpoints/fc-siam'),
                    restore_best=False)
    df = pd.read_csv(config.test)

    # do(config)
    evaluate(df, config)
