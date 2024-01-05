import matplotlib.pyplot as plt
import torch.cuda
import seaborn as sns

from .utils import train_loop
from torchsummary import summary
from torchview import draw_graph


class Compile:
    """
    compilation of the model
    """

    def __init__(self, model, optimizer, metrics=None, lr_scheduler=None):
        self.loss = None
        self.validation = None
        self.results = None
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler

        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        else:
            self.device = torch.device('cpu')

    def summary(self, input_size):
        summary(self.model.to(self.device), input_size=input_size, batch_size=-1)

    def visual_graph(self, input_size):
        model_graph = draw_graph(self.model.to(self.device), input_size=input_size, expand_nested=True)

        return model_graph.visual_graph

    def train(self, train_dataloader, loss, epochs=1, val_dataloader=None, multi_in=False, multi_out=False):
        self.validation = True if val_dataloader is not None else False
        self.loss = loss
        self.results = train_loop(model=self.model,
                                  loss_func=self.loss,
                                  train_loader=train_dataloader,
                                  test_loader=None,
                                  val_loader=val_dataloader,
                                  score_funcs=self.metrics,
                                  epochs=epochs,
                                  device=self.device,
                                  optimizer=self.optimizer,
                                  lr_schedule=self.lr_scheduler,
                                  multi_in=multi_in,
                                  multi_out=multi_out
                                  )
        return self.results

    def training_performance(self):
        plt.figure(figsize=(12, 6), dpi=200)
        sns.lineplot(x='epoch', y='train loss', data=self.results, label='train')

        if self.validation:
            sns.lineplot(x='epoch', y='val loss', data=self.results, label='validation')

        plt.ylabel('loss')
        plt.show()
