from notebooks.r_and_d.net_trainer import *
from forest_cover_change_detection.models.fc_siam import FCSiam

if __name__ == "__main__":
    config = Config('../../../data/annotated/',
                    '../../../data/train.csv',
                    '../../../data/annotated/test.csv',
                    FCSiam(3, 2, True),
                    nn.NLLLoss,
                    100,
                    32,
                    multi_in=True,
                    concat=False,
                    restore_best=True)
    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
