from notebooks.daudt.net_trainer import *
from forest_cover_change_detection.models.fc_ef import FCFE

if __name__ == "__main__":
    config = Config('../../../data/annotated/',
                    '../../../data/train.csv',
                    '../../../data/annotated/test.csv',
                    FCFE(6, 3),
                    nn.NLLLoss,
                    100,
                    32,
                    restore_best=True,
                    concat=True)
    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
