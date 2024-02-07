import torch.nn as nn
import torch
import torchvision.models as models
from PAR.multi_task_ddp import DMTPAR, DMTLoss
from PAR.par_utils import MTImageDataset
from torch.utils.data import ConcatDataset


def train_one_epoch(train_loader,optimizer,model, loss_fn):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        upper_color,lower_color,gender,bag_presence ,hat_presence  = model(inputs)
        loss = loss_fn([upper_color,lower_color,gender,bag_presence ,hat_presence ], labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 20 == 19:
            last_loss = running_loss / 20 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss


def train(epochs,LR = 10 ** -3, early_stopping = 3) -> None:
    f = open('./raports/resnet34_without_clahe_and_without_par_test1.txt','w+')
    criterion = DMTLoss()
    model = DMTPAR()
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),lr = LR)
    transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()
    training_set_data = MTImageDataset('./data/par_datasets/training_set.txt','./data/par_datasets/training_set' ,transform=transform)
    validation_set_data = MTImageDataset('./data/par_datasets/validation_set.txt','./data/par_datasets/validation_set' ,transform=transform)
    train_data = ConcatDataset([training_set_data, validation_set_data])
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64)
    model.train(True)
    prev_loss = 0
    count = 0
    print('START TRAINING')
    f.write('START TRAINING\n')
    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}')
        f.write(f'EPOCH {epoch + 1}\n')
        epoch_loss = train_one_epoch(train_loader,optimizer,model,criterion)
        print(f'LOSS: {epoch_loss}')
        f.write(f'LOSS: {epoch_loss}\n')
        torch.save(model.state_dict(),'./weights/resnet34_without_clahe_and_without_par/general_test1_' + str(epoch) + '.pt')
        if abs(epoch_loss-prev_loss) < 0.01:
            count = count + 1
            if count > early_stopping:
                break
        else:
            count = 0
        prev_loss = epoch_loss
    print('TRAINING FINISHED')
    f.write('TRAINING FINISHED')
    torch.save(model.state_dict(),'./weights/resnet34_without_clahe_and_without_par/general_test1.pt')
    ### TRAINING ON ATRIO CUES IMAGES ###
    new_train_data = MTImageDataset('./data/par_datasets/training_set_atrio_cues.txt','./data/par_datasets/training_set_atrio_cues' ,transform=transform)
    
    new_train_loader = torch.utils.data.DataLoader(new_train_data,batch_size=8)
    for group in optimizer.param_groups:
        group['lr'] /= 10
    prev_loss = 0
    count = 0
    print('START TRAINING ATRIO CUES')
    f.write('START TRAINING ATRIO CUES\n')
    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}')
        f.write(f'EPOCH {epoch + 1}\n')
        epoch_loss = train_one_epoch(new_train_loader,optimizer,model,criterion)
        print(f'LOSS: {epoch_loss}')
        f.write(f'LOSS: {epoch_loss}\n')
        torch.save(model.state_dict(),'./weights/resnet34_without_clahe_and_without_par/specific_test1_' + str(epoch) + '.pt')
        if abs(epoch_loss-prev_loss) < 0.01:
            count = count + 1
            if count > early_stopping:
                break
        else:
            count = 0
        prev_loss = epoch_loss
        print("Count:",count)
    print('TRAINING FINISHED ATRIO CUES')
    f.write('TRAINING FINISHED ATRIO CUES')
    f.close()
    torch.save(model.state_dict(),'./weights/resnet34_without_clahe_and_without_par/specific_test1.pt')


if __name__ == '__main__':
    train(17)