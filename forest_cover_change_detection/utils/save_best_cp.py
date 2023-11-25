import torch


class SaveBestCheckPoint:
    """save currently the best checkpoint of the model"""

    def __init__(self, path):
        self.best_val_score = 0
        self.path = path

    def __call__(self, current_score, epoch, model, optimizer, *args, **kwargs):
        if self.best_val_score < current_score:
            self.best_val_score = current_score

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': self.best_val_score
            }, f'{self.path}/best_model.pth')
