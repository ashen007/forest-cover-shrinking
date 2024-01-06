from torch import nn
from runs.r_and_d.net_trainer import *
from forest_cover_change_detection.models.v2.experimental.ex_7.model import AttentionEx1

if __name__ == "__main__":
    """test channel attention inside gate"""

    config = Config('../../../../../data/annotated/',
                    '../../../../../data/annotated/train.csv',
                    '../../../../../data/annotated/test.csv',
                    AttentionEx1(3, 2),
                    nn.NLLLoss,
                    300,
                    24,
                    restore_best=True,
                    concat=False,
                    multi_in=True,
                    patched=False)

    df = pd.read_csv(config.test)

    # do(config)
    evaluate(df, config)
