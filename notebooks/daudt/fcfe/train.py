import pandas as pd

from torch import nn
from ..net_trainer import do, evaluate, Config
from forest_cover_change_detection.models.fc_ef import FCFE
from forest_cover_change_detection.utils.save_best_cp import SaveBestCheckPoint

if __name__ == "__main__":
    config = Config(FCFE(6, 3, 2),
                    nn.NLLLoss,
                    100,
                    32,
                    checkpointer=SaveBestCheckPoint('../../checkpoints/fcfe'))
    df = pd.read_csv(config.test)

    do(config)
    evaluate(config.model, df)
