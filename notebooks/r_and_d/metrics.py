import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def metric_summary(root, renames):
    comp_df = []

    for sub in os.listdir(root):
        sub_dir = os.path.join(root, sub)

        if os.path.isdir(sub_dir):
            if sub.startswith('fc'):
                metric_file = os.path.join(sub_dir, 'metric_eval.csv')
                sub_metrics = pd.read_csv(metric_file)
                comp_df.append(sub_metrics.mean())

    comp_df = pd.concat(comp_df, axis=1)
    comp_df = comp_df.rename(columns=renames).T
    comp_df['f1'] = comp_df.apply(lambda x: 2 * ((x['precision'] * x['recall']) / (x['precision'] + x['recall'])),
                                  axis=1)

    return comp_df


def training_summary(root):
    train_loss_df = pd.DataFrame()
    val_loss_df = pd.DataFrame()

    for sub in os.listdir(root):
        sub_dir = os.path.join(root, sub)

        if os.path.isdir(sub_dir):
            if sub.startswith('fc'):
                metric_file = os.path.join(sub_dir, 'results.csv')
                sub_metrics = pd.read_csv(metric_file).rename(
                    columns={'train loss': f"t_{sub}",
                             'val loss': f'v_{sub}'})

                train_loss_df = pd.concat((train_loss_df,
                                           sub_metrics[f't_{sub}'])
                                          , axis=1)
                val_loss_df = pd.concat((val_loss_df,
                                         sub_metrics[f'v_{sub}'])
                                        , axis=1)

    return train_loss_df, val_loss_df


def training_curves(t, v, labels):
    plt.figure(figsize=(24, 12), dpi=300)
    plt.subplot(1, 2, 1)
    sns.lineplot(data=t, dashes=False,
                 palette=sns.color_palette('BrBG', 9), ci=None)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.legend(labels)

    plt.subplot(1, 2, 2)
    sns.lineplot(data=v, dashes=False,
                 palette=sns.color_palette('BrBG', 9), ci=None)
    plt.xlabel('epochs')
    plt.ylabel('val loss')
    plt.legend(labels)
    plt.show()


def model_heatmap(df, labels):
    plt.figure(figsize=(12, 6), dpi=200)
    g = sns.heatmap(df,
                    annot=True,
                    cbar=False,
                    cmap=sns.color_palette('BrBG'))
    g.set_yticklabels(labels)
    g.set_xticklabels(['overall acc', 'chng acc', 'no chng acc',
                       'precision', 'recall', 'dice', 'kappa', 'f-1'])
    plt.show()
