"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import torch
from tqdm import tqdm
import sys
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
sys.path.append(".")

from utils import AverageMeter, calculate_accuracy, calculate_accuracy_singleLabel


def train_epoch(epoch, data_set, model, criterion, optimizer, opt, logger, extra_train=False):
    print('train at epoch {}'.format(epoch))
    # scaler = GradScaler()
# batch_size = 4
    # gradient_accumulations = 4
# this means training will be done for affective batch size of 4 * 16 = 64


    model.train()

    losses = AverageMeter()
    WT_dice = AverageMeter()
    TC_dice = AverageMeter()
    ET_dice = AverageMeter()

    # data_set.file_open()
    train_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                               batch_size=opt["batch_size"], 
                                               shuffle=True, 
                                               pin_memory=True)
    training_process = tqdm(train_loader)
    for i, (inputs, targets) in enumerate(training_process):
        if i > 0:
            training_process.set_description("Epoch:%d;Loss:%.4f; dice-WT:%.4f, TC:%.4f, ET:%.4f, lr: %.6f"%(epoch,
                                             losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(),
                                             ET_dice.avg.item(), optimizer.param_groups[0]['lr']))

        if opt["cuda_devices"] is not None:
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        # optimizer.zero_grad()
        if opt["VAE_enable"]:
              # with autocast():

              # output = model(input)
              # loss = loss_fn(output, target)
              # print(f"input shape: {inputs.shape}")
              # print(f"Model er age:{targets.shape}")
              outputs, distr = model(inputs)
              # print(f"Outputs_shape: {outputs.shape}")
              loss = criterion(outputs, targets, distr)
        else:
              # with autocast():

              outputs = model(inputs)
              loss = criterion(outputs, targets)

        if opt["flooding"]:
            b = opt["flooding_level"]
            loss = (loss - b).abs() + b  # flooding

        if not opt["seg_dice"]:
            acc = calculate_accuracy(outputs.cpu(), targets.cpu())  # dice_coefficient
        else:
            acc = dict()
            acc["dice_wt"] = torch.tensor(0)
            acc["dice_tc"] = torch.tensor(0)
            acc["dice_et"] = torch.tensor(0)
            singleLabel_acc = calculate_accuracy_singleLabel(outputs.cpu(), targets.cpu())
            acc[opt["seg_dice"]] = singleLabel_acc

        losses.update(loss.cpu(), inputs.size(0))  # batch_avg
        WT_dice.update(acc["dice_wt"], inputs.size(0))
        TC_dice.update(acc["dice_tc"], inputs.size(0))
        ET_dice.update(acc["dice_et"], inputs.size(0))

        # optimizer.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        # scaler.scale(loss).backward()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        # scaler.scale(loss / gradient_accumulations).backward()
        # if (i + 1) % gradient_accumulations == 0:
        #   scaler.step(optimizer)
        #   scaler.update()
        #   model.zero_grad()


    logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'wt-dice': format(WT_dice.avg.item(), '.4f'),
        'tc-dice': format(TC_dice.avg.item(), '.4f'),
        'et-dice': format(ET_dice.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })

    if extra_train:
        return losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(), ET_dice.avg.item()


#attempt 

"""
# @author: Chenggang
# @github: https://github.com/MissShihongHowRU
# @time: 2020-09-09 22:04
# """
# import torch
# from tqdm import tqdm
# import sys
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# sys.path.append(".")

# from utils import AverageMeter, calculate_accuracy, calculate_accuracy_singleLabel


# def train_epoch(epoch, data_set, model, criterion, optimizer, opt, logger, extra_train=False):
#     print('train at epoch {}'.format(epoch))
#     # scaler = GradScaler()
# # batch_size = 4
#     # gradient_accumulations = 4
# # this means training will be done for affective batch size of 4 * 16 = 64


#     model.train()

#     losses = AverageMeter()
#     WT_dice = AverageMeter()
#     TC_dice = AverageMeter()
#     ET_dice = AverageMeter()

#     # data_set.file_open()
#     train_loader = torch.utils.data.DataLoader(dataset=data_set, 
#                                                batch_size=opt["batch_size"], 
#                                                shuffle=True, 
#                                                pin_memory=True)
#     training_process = tqdm(train_loader)
#     for i, (inputs, targets) in enumerate(training_process):
#         if i > 0:
#             training_process.set_description("Epoch:%d;Loss:%.4f; dice-WT:%.4f, TC:%.4f, ET:%.4f, lr: %.6f"%(epoch,
#                                              losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(),
#                                              ET_dice.avg.item(), optimizer.param_groups[0]['lr']))

#         if opt["cuda_devices"] is not None:
#             inputs = inputs.type(torch.FloatTensor)
#             inputs = inputs.cuda()
#             targets = targets.type(torch.FloatTensor)
#             targets = targets.cuda()
#         # optimizer.zero_grad()
#         if opt["VAE_enable"]:
#         #       with autocast():

#               # output = model(input)
#               # loss = loss_fn(output, target)
              
#             outputs, distr = model(inputs)
#             loss = criterion(outputs, targets, distr)
#         else:
#               # with autocast():

#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#         if opt["flooding"]:
#             b = opt["flooding_level"]
#             loss = (loss - b).abs() + b  # flooding

#         if not opt["seg_dice"]:
#             acc = calculate_accuracy(outputs.cpu(), targets.cpu())  # dice_coefficient
#         else:
#             acc = dict()
#             acc["dice_wt"] = torch.tensor(0)
#             acc["dice_tc"] = torch.tensor(0)
#             acc["dice_et"] = torch.tensor(0)
#             singleLabel_acc = calculate_accuracy_singleLabel(outputs.cpu(), targets.cpu())
#             acc[opt["seg_dice"]] = singleLabel_acc

#         losses.update(loss.cpu(), inputs.size(0))  # batch_avg
#         WT_dice.update(acc["dice_wt"], inputs.size(0))
#         TC_dice.update(acc["dice_tc"], inputs.size(0))
#         ET_dice.update(acc["dice_et"], inputs.size(0))

#         # optimizer.zero_grad()
#         optimizer.zero_grad()
#         loss.backward()
#         # scaler.scale(loss).backward()
#         optimizer.step()
#         # scaler.step(optimizer)
#         # scaler.update()
#         # scaler.scale(loss / gradient_accumulations).backward()
#         # if (i + 1) % gradient_accumulations == 0:
#         #   scaler.step(optimizer)
#         #   scaler.update()
#         #   model.zero_grad()


#     logger.log(phase="train", values={
#         'epoch': epoch,
#         'loss': format(losses.avg.item(), '.4f'),
#         'wt-dice': format(WT_dice.avg.item(), '.4f'),
#         'tc-dice': format(TC_dice.avg.item(), '.4f'),
#         'et-dice': format(ET_dice.avg.item(), '.4f'),
#         'lr': optimizer.param_groups[0]['lr']
#     })

#     if extra_train:
#         return losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(), ET_dice.avg.item()


