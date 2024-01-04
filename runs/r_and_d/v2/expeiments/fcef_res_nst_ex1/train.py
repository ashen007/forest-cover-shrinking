from torch import nn
from runs.r_and_d.net_trainer import *
from forest_cover_change_detection.models.v2.experimental.ex_1.module import FCFEResSplitAttention

if __name__ == "__main__":
    """test channel attention inside gate"""

    config = Config('../../../../../data/annotated/',
                    '../../../../../data/annotated/train.csv',
                    '../../../../../data/annotated/test.csv',
                    FCFEResSplitAttention(3, type_='ch'),
                    nn.NLLLoss,
                    300,
                    32,
                    restore_best=True,
                    concat=False,
                    multi_in=True,
                    patched=False)

    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
