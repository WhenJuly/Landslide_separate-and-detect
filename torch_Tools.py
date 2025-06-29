import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import time
import torch.nn.functional as F

class WaveformDataset(Dataset):
    def __init__(self, X_train, Y_train):
        # 移动轴的目的是将数据转换为 (batch_size, height, width, channels) 格式
        self.X_train = np.moveaxis(X_train, 1, -1)
        self.Y_train = np.moveaxis(Y_train, 1, -1)

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        X_waveform = self.X_train[idx]
        Y_waveform = self.Y_train[idx]
        return X_waveform,  Y_waveform


class WaveformDataset_h5(Dataset):
    def __init__(self, annotations_file):
        self.hdf5_file = h5py.File(annotations_file, 'r')

    def __len__(self):
        return self.hdf5_file['X_train'].shape[0]

    def __getitem__(self, idx):
        X_waveform = self.hdf5_file['X_train'][idx]
        Y_waveform = self.hdf5_file['Y_train'][idx]
        return X_waveform, Y_waveform


# from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# PyTorch
# Try the new loss function
class Explained_Variance_Loss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Explained_Variance_Loss, self).__init__()

    def forward(self, inputs, targets):
        return torch.var(targets - inputs, dim=2, unbiased=True, keepdim=True) \
               / torch.var(inputs, dim=2, unbiased=True, keepdim=True)


def try_gpu(i=0):  # @save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# count total number of parameters of a model
def parameter_number(model):
    num_param = 0
    for parameter in model.parameters():
        # print(parameter)
        num_param += np.prod(parameter.shape)
    return num_param


def training_loop(train_dataloader, validate_dataloader, model, loss_fn, optimizer, scheduler, epochs, patience, device):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)

    for epoch in range(1, epochs + 1):
        # estimate time for each epoch
        starttime = time.time()

        # ======================= training =======================
        # initialize the model for training
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record training loss
            train_losses.append(loss.item())

        if scheduler is not None: # Adjust the learning rate
            scheduler.step() 
            
        # ======================= validating =======================
        # initialize the model for training
        model.eval()
        for X, y in validate_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # record validation loss
            valid_losses.append(loss.item())

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # print training/validation statistics
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}\n' +
                     f'time per epoch: {(time.time() - starttime):.3f} s')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses

#仅有MSE
# def training_loop_branches(train_dataloader, validate_dataloader, model, loss_fn, optimizer, scheduler, epochs
#                            , patience, device, minimum_epochs=None):
#
#     # to track the average training loss per epoch as the model trains
#     avg_train_losses = []
#     avg_train_losses1 = []  # earthquake average loss with epoch
#     avg_train_losses2 = []  # noise average loss with epoch
#
#     # to track the average validation loss per epoch as the model trains
#     avg_valid_losses = []
#     avg_valid_losses1 = []  # earthquake average loss with epoch
#     avg_valid_losses2 = []  # noise average loss with epoch
#
#     # initialize the early_stopping object
#     if patience is None: # dont apply early stopping
#         early_stopping = EarlyStopping(patience=1, verbose=False)
#     else:
#         early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)
#
#     for epoch in range(1, epochs + 1):
#         # estimate time for each epoch
#         starttime = time.time()
#
#         # to track the training loss as the model trains
#         train_losses = []
#         train_losses1 = []  # earthquake loss
#         train_losses2 = []  # noise loss
#
#         # to track the validation loss as the model trains
#         valid_losses = []
#         valid_losses1 = []  # earthquake loss
#         valid_losses2 = []  # noise loss
#
#         # ======================= training =======================
#         # initialize the model for training
#         model.train()
#         size = len(train_dataloader.dataset)
#         for batch, (X, y) in enumerate(train_dataloader):
#             # Compute prediction and loss
#             X, y = X.to(device), y.to(device)
#             pred1, pred2 = model(X)
#
#             loss1 = loss_fn(pred1, y)
#             loss2 = loss_fn(pred2, X - y)
#
#             loss = loss1 + loss2
#
#             # record training loss
#             train_losses.append(loss.item())
#             train_losses1.append(loss1.item())
#             train_losses2.append(loss2.item())
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if scheduler is not None: # Adjust the learning rate
#             scheduler.step()
#
#         # ======================= validating =======================
#         # initialize the model for training
#         model.eval()
#         for X, y in validate_dataloader:
#             X, y = X.to(device), y.to(device)
#             pred1, pred2 = model(X)
#             loss1 = loss_fn(pred1, y)
#             loss2 = loss_fn(pred2, X - y)
#
#             #loss = loss1 + loss2
#
#             # record validation loss
#             valid_losses.append(loss1.item() + loss2.item())
#             valid_losses1.append(loss1.item())
#             valid_losses2.append(loss2.item())
#
#         # calculate average loss over an epoch
#         # total loss
#         train_loss = np.average(train_losses)
#         valid_loss = np.average(valid_losses)
#         avg_train_losses.append(train_loss)
#         avg_valid_losses.append(valid_loss)
#
#         # earthquake waveform loss
#         train_loss1 = np.average(train_losses1)
#         valid_loss1 = np.average(valid_losses1)
#         avg_train_losses1.append(train_loss1)
#         avg_valid_losses1.append(valid_loss1)
#
#         # ambient noise waveform loss
#         train_loss2 = np.average(train_losses2)
#         valid_loss2 = np.average(valid_losses2)
#         avg_train_losses2.append(train_loss2)
#         avg_valid_losses2.append(valid_loss2)
#
#         # print training/validation statistics
#         epoch_len = len(str(epochs))
#         print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
#                      f'train_loss: {train_loss:.5f} ' +
#                      f'valid_loss: {valid_loss:.5f}\n' +
#                      f'time per epoch: {(time.time() - starttime):.3f} s')
#
#         print(print_msg)
#
#         # clear lists to track next epoch
#         train_losses = []
#         valid_losses = []
#
#         if patience is not None:
#             if (minimum_epochs is None) or ((minimum_epochs is not None) and (epoch > minimum_epochs)):
#                 # early_stopping needs the validation loss to check if it has decresed,
#                 # and if it has, it will make a checkpoint of the current model
#                 early_stopping(valid_loss, model)
#
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#     # load the last checkpoint with the best model if apply early stopping
#     if patience is not None:
#         model.load_state_dict(torch.load('checkpoint.pt'))
#
#     partial_loss = [avg_train_losses1, avg_valid_losses1, avg_train_losses2, avg_valid_losses2]
#
#     return model, avg_train_losses, avg_valid_losses, partial_loss


def model_same(model1, model2):
    """Function to tell if two models are the same (same parameters)"""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        else:
            return True


# import torch
# import torch.nn.functional as F

#交叉熵
# def training_loop_branches(train_dataloader, validate_dataloader, model, optimizer, scheduler, epochs, patience, device,
#                            minimum_epochs=None):
#     # to track the average training loss per epoch as the model trains
#     avg_train_losses = []
#     avg_train_losses1 = []  # earthquake average loss with epoch
#     avg_train_losses2 = []  # noise average loss with epoch
#
#     # to track the average validation loss per epoch as the model trains
#     avg_valid_losses = []
#     avg_valid_losses1 = []  # earthquake average loss with epoch
#     avg_valid_losses2 = []  # noise average loss with epoch
#
#     # initialize the early_stopping object
#     if patience is None:  # don't apply early stopping
#         early_stopping = EarlyStopping(patience=1, verbose=False)
#     else:
#         early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)
#
#     for epoch in range(1, epochs + 1):
#         # estimate time for each epoch
#         starttime = time.time()
#
#         # to track the training loss as the model trains
#         train_losses = []
#         train_losses1 = []  # earthquake loss
#         train_losses2 = []  # noise loss
#
#         # to track the validation loss as the model trains
#         valid_losses = []
#         valid_losses1 = []  # earthquake loss
#         valid_losses2 = []  # noise loss
#
#         # ======================= training =======================
#         # initialize the model for training
#         model.train()
#         size = len(train_dataloader.dataset)
#         for batch, (X, y) in enumerate(train_dataloader):
#             # Compute prediction and loss
#             X, y = X.to(device), y.to(device)
#             pred1, pred2 = model(X)
#
#             # Ensure y and (X - y) are in the correct format (binary 0/1)
#             y = y.float()
#             target2 = (X - y).float()
#
#             # Use binary_cross_entropy_with_logits
#             loss1 = F.binary_cross_entropy_with_logits(pred1, y)
#             loss2 = F.binary_cross_entropy_with_logits(pred2, target2)
#
#             loss = loss1 + loss2
#
#             # record training loss
#             train_losses.append(loss.item())
#             train_losses1.append(loss1.item())
#             train_losses2.append(loss2.item())
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if scheduler is not None:  # Adjust the learning rate
#             scheduler.step()
#
#         # ======================= validating =======================
#         # initialize the model for training
#         model.eval()
#         for X, y in validate_dataloader:
#             X, y = X.to(device), y.to(device)
#             pred1, pred2 = model(X)
#             # Ensure y and (X - y) are in the correct format (binary 0/1)
#             y = y.float()
#             target2 = (X - y).float()
#
#             # Use binary_cross_entropy_with_logits
#             loss1 = F.binary_cross_entropy_with_logits(pred1, y)
#             loss2 = F.binary_cross_entropy_with_logits(pred2, target2)
#
#             #loss = loss1 + loss2
#
#             # record validation loss
#             valid_losses.append(loss1.item() + loss2.item())
#             valid_losses1.append(loss1.item())
#             valid_losses2.append(loss2.item())
#
#         # calculate average loss over an epoch
#         # total loss
#         train_loss = np.average(train_losses)
#         valid_loss = np.average(valid_losses)
#         avg_train_losses.append(train_loss)
#         avg_valid_losses.append(valid_loss)
#
#         # earthquake waveform loss
#         train_loss1 = np.average(train_losses1)
#         valid_loss1 = np.average(valid_losses1)
#         avg_train_losses1.append(train_loss1)
#         avg_valid_losses1.append(valid_loss1)
#
#         # ambient noise waveform loss
#         train_loss2 = np.average(train_losses2)
#         valid_loss2 = np.average(valid_losses2)
#         avg_train_losses2.append(train_loss2)
#         avg_valid_losses2.append(valid_loss2)
#
#         # print training/validation statistics
#         epoch_len = len(str(epochs))
#         print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
#                      f'train_loss: {train_loss:.5f} ' +
#                      f'valid_loss: {valid_loss:.5f}\n' +
#                      f'time per epoch: {(time.time() - starttime):.3f} s')
#
#         print(print_msg)
#
#         # clear lists to track next epoch
#         train_losses = []
#         valid_losses = []
#
#         if patience is not None:
#             if (minimum_epochs is None) or ((minimum_epochs is not None) and (epoch > minimum_epochs)):
#                 # early_stopping needs the validation loss to check if it has decresed,
#                 # and if it has, it will make a checkpoint of the current model
#                 early_stopping(valid_loss, model)
#
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#     # load the last checkpoint with the best model if apply early stopping
#     if patience is not None:
#         model.load_state_dict(torch.load('checkpoint.pt'))
#
#     partial_loss = [avg_train_losses1, avg_valid_losses1, avg_train_losses2, avg_valid_losses2]
#
#     return model, avg_train_losses, avg_valid_losses, partial_loss

# def combined_loss(pred1, pred2, y, x, alpha=0.5):
#     # pred1 is the predicted earthquake signal
#     # pred2 is the predicted noise signal
#     # y is the true earthquake signal
#     # x is the input signal (earthquake + noise)
#
#     # Mean Squared Error (MSE) loss
#     mse_loss1 = F.mse_loss(pred1, y)
#     mse_loss2 = F.mse_loss(pred2, x - y)
#
#     # Cross-Entropy (CE) loss
#     ce_loss1 = F.binary_cross_entropy_with_logits(pred1, y)
#     ce_loss2 = F.binary_cross_entropy_with_logits(pred2, x - y)
#
#     # Combined loss
#     loss1 = alpha * mse_loss1 + (1 - alpha) * ce_loss1
#     loss2 = alpha * mse_loss2 + (1 - alpha) * ce_loss2
#
#     return loss1 + loss2, mse_loss1, mse_loss2, ce_loss1, ce_loss2
#
# #MSE + 交叉熵
# def training_loop_branches(train_dataloader, validate_dataloader, model, optimizer, scheduler, epochs
#                            , patience, device, minimum_epochs=None):
#     # to track the average training loss per epoch as the model trains
#     avg_train_losses = []
#     avg_train_losses1 = []  # earthquake average loss with epoch
#     avg_train_losses2 = []  # noise average loss with epoch
#     avg_train_ce_losses1 = []  # earthquake CE loss with epoch
#     avg_train_ce_losses2 = []  # noise CE loss with epoch
#
#     # to track the average validation loss per epoch as the model trains
#     avg_valid_losses = []
#     avg_valid_losses1 = []  # earthquake average loss with epoch
#     avg_valid_losses2 = []  # noise average loss with epoch
#     avg_valid_ce_losses1 = []  # earthquake CE loss with epoch
#     avg_valid_ce_losses2 = []  # noise CE loss with epoch
#
#     # initialize the early_stopping object
#     if patience is None: # dont apply early stopping
#         early_stopping = EarlyStopping(patience=1, verbose=False)
#     else:
#         early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)
#
#     for epoch in range(1, epochs + 1):
#         # estimate time for each epoch
#         starttime = time.time()
#
#         # to track the training loss as the model trains
#         train_losses = []
#         train_losses1 = []  # earthquake loss
#         train_losses2 = []  # noise loss
#         train_ce_losses1 = []  # earthquake CE loss
#         train_ce_losses2 = []  # noise CE loss
#
#         # to track the validation loss as the model trains
#         valid_losses = []
#         valid_losses1 = []  # earthquake loss
#         valid_losses2 = []  # noise loss
#         valid_ce_losses1 = []  # earthquake CE loss
#         valid_ce_losses2 = []  # noise CE loss
#
#         # ======================= training =======================
#         # initialize the model for training
#         model.train()
#         size = len(train_dataloader.dataset)
#         for batch, (X, y) in enumerate(train_dataloader):
#             # Compute prediction and loss
#             X, y = X.to(device), y.to(device)
#             pred1, pred2 = model(X)
#             loss, mse_loss1, mse_loss2, ce_loss1, ce_loss2 = combined_loss(pred1, pred2, y, X)
#
#             # record training loss
#             train_losses.append(loss.item())
#             train_losses1.append(mse_loss1.item())
#             train_losses2.append(mse_loss2.item())
#             train_ce_losses1.append(ce_loss1.item())
#             train_ce_losses2.append(ce_loss2.item())
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if scheduler is not None: # Adjust the learning rate
#             scheduler.step()
#
#         # ======================= validating =======================
#         # initialize the model for validation
#         model.eval()
#         with torch.no_grad():
#             for X, y in validate_dataloader:
#                 X, y = X.to(device), y.to(device)
#                 pred1, pred2 = model(X)
#                 loss, mse_loss1, mse_loss2, ce_loss1, ce_loss2 = combined_loss(pred1, pred2, y, X)
#
#                 # record validation loss
#                 valid_losses.append(loss.item())
#                 valid_losses1.append(mse_loss1.item())
#                 valid_losses2.append(mse_loss2.item())
#                 valid_ce_losses1.append(ce_loss1.item())
#                 valid_ce_losses2.append(ce_loss2.item())
#
#         # calculate average loss over an epoch
#         # total loss
#         train_loss = np.average(train_losses)
#         valid_loss = np.average(valid_losses)
#         avg_train_losses.append(train_loss)
#         avg_valid_losses.append(valid_loss)
#
#         # earthquake waveform loss
#         train_loss1 = np.average(train_losses1)
#         valid_loss1 = np.average(valid_losses1)
#         avg_train_losses1.append(train_loss1)
#         avg_valid_losses1.append(valid_loss1)
#
#         # ambient noise waveform loss
#         train_loss2 = np.average(train_losses2)
#         valid_loss2 = np.average(valid_losses2)
#         avg_train_losses2.append(train_loss2)
#         avg_valid_losses2.append(valid_loss2)
#
#         # earthquake CE loss
#         train_ce_loss1 = np.average(train_ce_losses1)
#         valid_ce_loss1 = np.average(valid_ce_losses1)
#         avg_train_ce_losses1.append(train_ce_loss1)
#         avg_valid_ce_losses1.append(valid_ce_loss1)
#
#         # ambient noise CE loss
#         train_ce_loss2 = np.average(train_ce_losses2)
#         valid_ce_loss2 = np.average(valid_ce_losses2)
#         avg_train_ce_losses2.append(train_ce_loss2)
#         avg_valid_ce_losses2.append(valid_ce_loss2)
#
#         # print training/validation statistics
#         epoch_len = len(str(epochs))
#         print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
#                      f'train_loss: {train_loss:.5f} ' +
#                      f'valid_loss: {valid_loss:.5f}\n' +
#                      f'time per epoch: {(time.time() - starttime):.3f} s')
#
#         print(print_msg)
#
#         # clear lists to track next epoch
#         train_losses = []
#         valid_losses = []
#
#         if patience is not None:
#             if (minimum_epochs is None) or ((minimum_epochs is not None) and (epoch > minimum_epochs)):
#                 # early_stopping needs the validation loss to check if it has decreased,
#                 # and if it has, it will make a checkpoint of the current model
#                 early_stopping(valid_loss, model)
#
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#     # load the last checkpoint with the best model if apply early stopping
#     if patience is not None:
#         model.load_state_dict(torch.load('checkpoint.pt'))
#
#     partial_loss = [avg_train_losses1, avg_valid_losses1, avg_train_losses2, avg_valid_losses2]
#
#     return model, avg_train_losses, avg_valid_losses, partial_loss

#********************
# def pearson_corrcoef(x, y):
#     mean_x = torch.mean(x)
#     mean_y = torch.mean(y)
#     xm = x - mean_x
#     ym = y - mean_y
#     r_num = torch.sum(xm * ym)
#     r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
#     r = r_num / r_den
#     return r
#
# def combined_loss(pred1, pred2, y, alpha=0.7):
#     # pred1 is the predicted earthquake signal
#     # pred2 is the predicted noise signal
#     # y is the true earthquake signal
#     # x is the input signal (earthquake + noise)
#
#     # Mean Squared Error (MSE) loss
#     mse_loss1 = F.mse_loss(pred1, y)
#     mse_loss2 = F.mse_loss(pred2, 1 - y)
#
#     # Pearson Correlation Coefficient loss
#     cc_loss1 = 1 - pearson_corrcoef(pred1, y)
#     cc_loss2 = 1 - pearson_corrcoef(pred2, 1 - y) #这两个标签相对于0.5
#
#     # Combined loss
#     loss1 = alpha * mse_loss1 + (1-alpha)*cc_loss1
#     loss2 = alpha * mse_loss2 + (1-alpha)*cc_loss2
#
#     return loss1 + loss2, mse_loss1, mse_loss2, cc_loss1, cc_loss2
#
# def training_loop_branches(train_dataloader, validate_dataloader, model, optimizer, scheduler, epochs, patience, device, minimum_epochs=None):
#     # to track the average training loss per epoch as the model trains
#     avg_train_losses = []
#     avg_train_losses1 = []  # earthquake average loss with epoch
#     avg_train_losses2 = []  # noise average loss with epoch
#     avg_train_cc_losses1 = []  # earthquake CC loss with epoch
#     avg_train_cc_losses2 = []  # noise CC loss with epoch
#
#     # to track the average validation loss per epoch as the model trains
#     avg_valid_losses = []
#     avg_valid_losses1 = []  # earthquake average loss with epoch
#     avg_valid_losses2 = []  # noise average loss with epoch
#     avg_valid_cc_losses1 = []  # earthquake CC loss with epoch
#     avg_valid_cc_losses2 = []  # noise CC loss with epoch
#
#     # initialize the early_stopping object
#     if patience is None: # don't apply early stopping
#         early_stopping = EarlyStopping(patience=1, verbose=False)
#     else:
#         early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)
#
#     for epoch in range(1, epochs + 1):
#         # estimate time for each epoch
#         starttime = time.time()
#
#         # to track the training loss as the model trains
#         train_losses = []
#         train_losses1 = []  # earthquake loss
#         train_losses2 = []  # noise loss
#         train_cc_losses1 = []  # earthquake CC loss
#         train_cc_losses2 = []  # noise CC loss
#
#         # to track the validation loss as the model trains
#         valid_losses = []
#         valid_losses1 = []  # earthquake loss
#         valid_losses2 = []  # noise loss
#         valid_cc_losses1 = []  # earthquake CC loss
#         valid_cc_losses2 = []  # noise CC loss
#
#         # ======================= training =======================
#         # initialize the model for training
#         model.train()
#         size = len(train_dataloader.dataset)
#         for batch, (X, y) in enumerate(train_dataloader):
#             # Compute prediction and loss
#             X, y = X.to(device), y.to(device)
#             pred1, pred2 = model(X)
#             loss, mse_loss1, mse_loss2, cc_loss1, cc_loss2 = combined_loss(pred1, pred2, y)
#
#             # record training loss
#             train_losses.append(loss.item())
#             train_losses1.append(mse_loss1.item())
#             train_losses2.append(mse_loss2.item())
#             train_cc_losses1.append(cc_loss1.item())
#             train_cc_losses2.append(cc_loss2.item())
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if scheduler is not None: # Adjust the learning rate
#             scheduler.step()
#
#         # ======================= validating =======================
#         # initialize the model for validation
#         model.eval()
#         with torch.no_grad():
#             for X, y in validate_dataloader:
#                 X, y = X.to(device), y.to(device)
#                 pred1, pred2 = model(X)
#                 loss, mse_loss1, mse_loss2, cc_loss1, cc_loss2 = combined_loss(pred1, pred2, y)
#
#                 # record validation loss
#                 valid_losses.append(loss.item())
#                 valid_losses1.append(mse_loss1.item())
#                 valid_losses2.append(mse_loss2.item())
#                 valid_cc_losses1.append(cc_loss1.item())
#                 valid_cc_losses2.append(cc_loss2.item())
#
#         # calculate average loss over an epoch
#         # total loss
#         train_loss = np.average(train_losses)
#         valid_loss = np.average(valid_losses)
#         avg_train_losses.append(train_loss)
#         avg_valid_losses.append(valid_loss)
#
#         # earthquake waveform loss
#         train_loss1 = np.average(train_losses1)
#         valid_loss1 = np.average(valid_losses1)
#         avg_train_losses1.append(train_loss1)
#         avg_valid_losses1.append(valid_loss1)
#
#         # ambient noise waveform loss
#         train_loss2 = np.average(train_losses2)
#         valid_loss2 = np.average(valid_losses2)
#         avg_train_losses2.append(train_loss2)
#         avg_valid_losses2.append(valid_loss2)
#
#         # earthquake CC loss
#         train_cc_loss1 = np.average(train_cc_losses1)
#         valid_cc_loss1 = np.average(valid_cc_losses1)
#         avg_train_cc_losses1.append(train_cc_loss1)
#         avg_valid_cc_losses1.append(valid_cc_loss1)
#
#         # ambient noise CC loss
#         train_cc_loss2 = np.average(train_cc_losses2)
#         valid_cc_loss2 = np.average(valid_cc_losses2)
#         avg_train_cc_losses2.append(train_cc_loss2)
#         avg_valid_cc_losses2.append(valid_cc_loss2)
#
#         # print training/validation statistics
#         epoch_len = len(str(epochs))
#         print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
#                      f'train_loss: {train_loss:.5f} ' +
#                      f'valid_loss: {valid_loss:.5f}\n' +
#                      f'time per epoch: {(time.time() - starttime):.3f} s')
#
#         print(print_msg)
#
#         # clear lists to track next epoch
#         train_losses = []
#         valid_losses = []
#
#         if patience is not None:
#             if (minimum_epochs is None) or ((minimum_epochs is not None) and (epoch > minimum_epochs)):
#                 # early_stopping needs the validation loss to check if it has decreased,
#                 # and if it has, it will make a checkpoint of the current model
#                 early_stopping(valid_loss, model)
#
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#     # load the last checkpoint with the best model if apply early stopping
#     if patience is not None:
#         model.load_state_dict(torch.load('checkpoint.pt'))
#
#     partial_loss = [avg_train_losses1, avg_valid_losses1, avg_train_losses2, avg_valid_losses2]
#
#     return model, avg_train_losses, avg_valid_losses, partial_loss

#DWA
# def training_loop_branches(train_dataloader, validate_dataloader, model, optimizer, scheduler, epochs, patience, device,
#                            minimum_epochs=None):
#     avg_train_losses = []
#     avg_valid_losses = []
#
#     if patience is None:
#         early_stopping = EarlyStopping(patience=1, verbose=False)
#     else:
#         early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)
#
#     # Initialize dynamic weights and average task loss arrays
#     loss_weights = np.ones((epochs + 1, 2))  # +1 to account for index start at 1
#     avg_task_loss = np.zeros((epochs + 1, 2))
#     T = 2.0  # Temperature parameter for DWA (adjustable)
#
#     for epoch in range(1, epochs + 1):
#         starttime = time.time()
#
#         # Adjust weights based on DWA strategy
#         if epoch == 1:
#             loss_weights[epoch] = [1.0, 1.0]
#         else:
#             sum_w = []
#             for i in range(2):
#                 if avg_task_loss[epoch - 2, i] > 0:
#                     w = avg_task_loss[epoch - 1, i] / avg_task_loss[epoch - 2, i]
#                 else:
#                     w = 1.0
#                 sum_w.append(np.exp(w / T))
#             loss_weights[epoch] = [2 * w / np.sum(sum_w) for w in sum_w]
#
#         # Record training and validation losses
#         train_losses = []
#         valid_losses = []
#
#         model.train()
#         epoch_trn_multi_loss = np.zeros(2)  # To store loss1 and loss2 for the current epoch
#
#         for X, y1, y2 in train_dataloader:  # y1: separation labels, y2: recognition labels
#             X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
#
#             pred1, pred2 = model(X)
#
#             # Compute loss functions
#             loss1 = combined_loss(pred1, y1)
#             loss_CE = torch.nn.CrossEntropyLoss()
#             loss2 = loss_CE(pred2, y2)
#
#             # Apply DWA weights to losses
#             weighted_loss1 = loss_weights[epoch, 0] * loss1
#             weighted_loss2 = loss_weights[epoch, 1] * loss2
#             loss = weighted_loss1 + weighted_loss2
#
#             # Record loss
#             train_losses.append(loss.item())
#             epoch_trn_multi_loss[0] += loss1.item()
#             epoch_trn_multi_loss[1] += loss2.item()
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         # Calculate average task loss for the epoch
#         avg_task_loss[epoch] = epoch_trn_multi_loss / len(train_dataloader)
#
#         print(f'Epoch {epoch} - Training L1: {loss1:.6f} (Weighted: {weighted_loss1:.6f}), Training L2: {loss2:.6f} (Weighted: {weighted_loss2:.6f})')
#
#         if scheduler is not None:
#             scheduler.step()
#
#         model.eval()
#         with torch.no_grad():
#             epoch_val_multi_loss = np.zeros(2)  # To store validation loss1 and loss2 for the current epoch
#             for X, y1, y2 in validate_dataloader:
#                 X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
#                 pred1, pred2 = model(X)
#
#                 loss1 = combined_loss(pred1, y1)
#                 loss_CE = torch.nn.CrossEntropyLoss()
#                 loss2 = loss_CE(pred2, y2)
#
#                 weighted_loss1 = loss_weights[epoch, 0] * loss1
#                 weighted_loss2 = loss_weights[epoch, 1] * loss2
#                 loss = weighted_loss1 + weighted_loss2
#
#                 valid_losses.append(loss.item())
#                 epoch_val_multi_loss[0] += loss1.item()
#                 epoch_val_multi_loss[1] += loss2.item()
#
#             print(f'Epoch {epoch} - Validation L1: {loss1:.6f} (Weighted: {weighted_loss1:.6f}), Validation L2: {loss2:.6f} (Weighted: {weighted_loss2:.6f})')
#
#         avg_train_losses.append(np.average(train_losses))
#         avg_valid_losses.append(np.average(valid_losses))
#
#         print_msg = (f'[{epoch:>{len(str(epochs))}}/{epochs}] ' +
#                      f'train_loss: {np.average(train_losses):.5f} ' +
#                      f'valid_loss: {np.average(valid_losses):.5f}\n' +
#                      f'time per epoch: {(time.time() - starttime):.3f} s')
#         print(print_msg)
#
#         if patience is not None:
#             if (minimum_epochs is None) or ((minimum_epochs is not None) and (epoch > minimum_epochs)):
#                 early_stopping(np.average(valid_losses), model)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#     # Ensure the model is restored if early stopping occurred
#     if patience is not None and early_stopping.early_stop:
#         model.load_state_dict(torch.load('checkpoint.pt'))
#
#     return model, avg_train_losses, avg_valid_losses

# only CrossEntropyLoss
def training_loop_branches(train_dataloader, validate_dataloader, model, optimizer, scheduler, epochs, patience, device,
                           minimum_epochs=None):
    avg_train_losses = []
    avg_valid_losses = []

    if patience is None:
        early_stopping = EarlyStopping(patience=1, verbose=False)
    else:
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)

    for epoch in range(1, epochs + 1):
        starttime = time.time()

        # Record training and validation losses
        train_losses = []
        valid_losses = []

        model.train()
        for X, y in train_dataloader:  # y1: separation labels, y2: recognition labels
            X, y = X.to(device), y.to(device)

            # y2 = y2.long()
            pred1 = model(X)
            # print(pred2.shape)
            # print(y2.shape)
            # pred2 = pred2.squeeze(1)

            # Compute loss functions
            # loss2 = F.cross_entropy(pred2, y2)
            loss_CE = torch.nn.CrossEntropyLoss()
            loss = loss_CE(pred1, y)
            # print(loss2.shape)
            # loss = loss

            # Record loss
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Training L is : {loss}')
        # print(f'Training L2 is : {loss2}')

        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            for X, y1 in validate_dataloader:
                X, y1 = X.to(device), y1.to(device)
                pred1 = model(X)
                # pred2 = pred2.squeeze(1)

                loss_CE = torch.nn.CrossEntropyLoss()
                loss = loss_CE(pred1, y1)

                valid_losses.append(loss.item())

            print(f'validating L is : {loss}')

        avg_train_losses.append(np.average(train_losses))
        avg_valid_losses.append(np.average(valid_losses))

        print_msg = (f'[{epoch:>{len(str(epochs))}}/{epochs}] ' +
                     f'train_loss: {np.average(train_losses):.5f} ' +
                     f'valid_loss: {np.average(valid_losses):.5f}\n' +
                     f'time per epoch: {(time.time() - starttime):.3f} s')
        print(print_msg)

        if patience is not None:
            if (minimum_epochs is None) or ((minimum_epochs is not None) and (epoch > minimum_epochs)):
                early_stopping(np.average(valid_losses), model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    if patience is not None:
        model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses
