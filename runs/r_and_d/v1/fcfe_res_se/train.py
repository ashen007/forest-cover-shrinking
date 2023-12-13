from runs.r_and_d.net_trainer import *
from forest_cover_change_detection.models.v1.fc_ef_res_se import FCFEResSE

if __name__ == "__main__":
    config = Config('../../../data/annotated/',
                    '../../../data/subsets/256/train.csv',
                    '../../../data/annotated/test.csv',
                    FCFEResSE(6, 2),
                    nn.NLLLoss,
                    100,
                    32,
                    restore_best=True,
                    concat=True)
    df = pd.read_csv(config.test)

    # do(config)
    evaluate(df, config)
