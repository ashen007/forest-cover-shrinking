from notebooks.r_and_d.net_trainer import *
from forest_cover_change_detection.models.fcfe_with_att.fcfe_att import FCFEWithAttention

if __name__ == "__main__":
    config = Config('../../../data/annotated/',
                    '../../../data/subsets/256/train.csv',
                    '../../../data/annotated/test.csv',
                    FCFEWithAttention(6, 2),
                    nn.NLLLoss,
                    100,
                    32,
                    restore_best=True,
                    concat=True)
    df = pd.read_csv(config.test)

    do(config)
    evaluate(df, config)
