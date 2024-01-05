import time
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from forest_cover_change_detection.utils.move import move_to


def get_scores(*, model, optimizer, epoch,
               y_true, y_pred, score_funcs,
               prefix, results, result_str):
    """
    get scores for given score functions and add results to
    a result object, and save-based checkpoint based on a result from
    score function results getting worse or getting better

    :param model:
    :param optimizer:
    :param epoch:
    :param y_true:
    :param y_pred:
    :param score_funcs:
    :param prefix:
    :param results:
    :param result_str:
    :return:
    """
    for name, score_func in score_funcs.items():
        try:
            score = score_func(y_true, y_pred)

            results[prefix + " " + name].append(score)
            result_str.append(f"{prefix} {name}: {score}")

            # if prefix == 'val' or prefix == 'test':
            #     checkpointer(score, (epoch + 1), model, optimizer)

        except:
            results[prefix + " " + name].append(float("NaN"))
            result_str.append(f"{prefix} {name}: {float('NaN')}")


def do_training(*, model, optimizer, inputs, labels, loss_func, multi_out=False):
    if multi_out:
        y_hats = model(inputs)
        loss = 0
        y_hat = y_hats[0.55]

        for k, v in y_hats.items():
            loss_ = loss_func(v, labels) * (2 ** -(k))
            loss += loss_
    else:
        y_hat = model(inputs)  # this just computed f_Θ(x(i))
        loss = loss_func(y_hat, labels)  # Compute loss.

    if model.training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return y_hat, loss


def do_training_with_multiple_in(*, model, optimizer, inputs, labels, loss_func, multi_out=False):
    if multi_out:
        y_hats = model(*inputs)
        loss = 0
        y_hat = y_hats[0.55]

        for k, v in y_hats.items():
            loss_ = loss_func(v, labels) * (2 ** -(k))
            loss += loss_
    else:
        y_hat = model(*inputs)  # this just computed f_Θ(x(i))
        loss = loss_func(y_hat, labels)  # Compute loss.

    if model.training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return y_hat, loss


def run_epoch(model, optimizer, data_loader,
              loss_func, device, results,
              score_funcs, multi_in, prefix="",
              desc=None, epoch=None, multi_out=False):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()

    for inputs, labels in tqdm(data_loader, unit="steps"):
        # Move the batch to the device we are using.
        inputs = move_to(inputs, device)
        labels = move_to(labels, device)

        if not multi_in:
            y_hat, loss = do_training(model=model, optimizer=optimizer, inputs=inputs,
                                      labels=labels, loss_func=loss_func,
                                      multi_out=multi_out)

        else:
            y_hat, loss = do_training_with_multiple_in(model=model, optimizer=optimizer, inputs=inputs,
                                                       labels=labels, loss_func=loss_func,
                                                       multi_out=multi_out)

        # Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            # moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            # add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

    # end training epoch
    end = time.time()

    y_pred = np.asarray(y_pred)

    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:  # We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    # Else, we assume we are working on a regression problem

    results[prefix + " loss"].append(np.mean(running_loss))
    result_str = [f"{prefix} loss: {np.mean(running_loss)}"]

    get_scores(model=model, optimizer=optimizer, epoch=epoch, y_true=y_true,
               y_pred=y_pred, score_funcs=score_funcs, prefix=prefix,
               results=results, result_str=result_str)

    print(" ".join(result_str))

    return end - start  # time spent on epoch


def create_trackers(to_track, score_funcs, test_loader, val_loader):
    if val_loader is not None:
        to_track.append("val loss")

    if test_loader is not None:
        to_track.append("test loss")

    for eval_score in score_funcs:
        to_track.append("train " + eval_score)

        if val_loader is not None:
            to_track.append("val " + eval_score)

        if test_loader is not None:
            to_track.append("test " + eval_score)


def do_validation(model, optimizer, data_loader,
                  loss_func, device, results,
                  score_funcs, prefix, desc, epoch, multi_in,
                  multi_out):
    model = model.eval()

    with torch.no_grad():
        run_epoch(model, optimizer, data_loader, loss_func, device,
                  results, score_funcs, multi_in, prefix, desc, epoch,
                  multi_out)


def do_change_lr(lr_schedule, results):
    if lr_schedule is not None:
        if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_schedule.step(results["val loss"][-1])

        else:
            lr_schedule.step()


def save_checkpoint(epoch, model, optimizer, results, checkpoint_file):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results
    }, checkpoint_file)


def train_loop(model, loss_func,
               train_loader, test_loader=None, val_loader=None,
               score_funcs=None, epochs=50, device="cpu",
               optimizer=None, lr_schedule=None,
               checkpoint_file="last-checkpoint.pth", keep_best=True,
               multi_in=False, multi_out=False):
    if score_funcs is None:
        score_funcs = {}

    to_track = ["epoch", "total time", "train loss", "lr"]

    create_trackers(to_track, score_funcs, test_loader, val_loader)

    total_train_time = 0  # How long have we spent in the training loop?
    results = {}
    current_best = 100000

    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        del_opt = True

    else:
        del_opt = False

    # Place the model on the correct computed resource (CPU or GPU)
    model.to(device)

    for epoch in range(epochs):
        model = model.train()  # Put our model in training mode

        # epoch number
        print(f"Epoch: {epoch}/{epochs}")

        run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs,
                  prefix="train", desc="Training", multi_in=multi_in, multi_out=multi_out)

        results["total time"].append(total_train_time)
        results["epoch"].append(epoch)
        results["lr"].append(optimizer.param_groups[0]['lr'])

        if val_loader is not None:
            do_validation(model, optimizer, val_loader, loss_func, device, results, score_funcs, prefix="val",
                          desc="Validation", epoch=epoch, multi_in=multi_in, multi_out=multi_out)

        if test_loader is not None:
            do_validation(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test",
                          desc="Testing", epoch=epoch, multi_in=multi_in, multi_out=multi_out)

        do_change_lr(lr_schedule, results)

        save_checkpoint(epoch, model, optimizer, results, checkpoint_file)

        if (current_best >= results["val loss"][-1]) and keep_best:
            current_best = results["val loss"][-1]
            save_checkpoint(epoch, model, optimizer, results, 'best_model.pth')

    if del_opt:
        del optimizer

    return pd.DataFrame.from_dict(results)
