import torch.nn as nn
import torch
import torchvision.models as models
from PAR.multi_task import AMTLoss,AMTPAR
from PAR.par_utils import MTImageDataset
from TOOLS.bg_remover import BgRemover


def save_epochs(epoch_nr,loss,model):
    torch.save(model.state_dict(),f'./weights/train0/multitask_{epoch_nr}_loss:{round(loss,4)}.pt')


def train_one_epoch(train_loader,optimizer,model,loss_fn):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to('cuda'), labels.to('cuda').unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 20 == 19:
            last_loss = running_loss / 20 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


def train(epochs,LR = 10 ** -3) -> None:
    f = open('./raports/train_multi_raport.txt','w+')
    criterion = AMTLoss()
    model = AMTPAR()
    optimizer = torch.optim.AdamW(model.parameters(),lr = LR)
    transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    train_data = MTImageDataset('./data/par_datasets/training_set.txt','./data/par_datasets/training_set' ,transform=transform,clahe=True)
    train_custom_data = MTImageDataset('./data/par_datasets/custom_set.txt','./data/par_datasets/custom_set' ,transform=transform,clahe=True)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64)
    custom_loader = torch.utils.data.DataLoader(train_custom_data,batch_size = 16)
    model.train(True)
    print('START TRAINING')
    f.write('START TRAINING\n')

    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}')
        f.write(f'EPOCH {epoch + 1}\n')
        epoch_loss = train_one_epoch(train_loader,optimizer,model,criterion)
        save_epochs(epoch,epoch_loss,model)
        print(f'LOSS: {epoch_loss}')
        f.write(f'LOSS: {epoch_loss}\n')

    for group in optimizer.param_groups:
        group['lr'] /= 80

    for epoch in range(epochs,2*epochs + 1):
        print(f'EPOCH {epoch + 1}')
        f.write(f'EPOCH {epoch + 1}\n')
        epoch_loss = train_one_epoch(custom_loader,optimizer,model,criterion)
        save_epochs(epoch,epoch_loss,model)
        print(f'LOSS: {epoch_loss}')
        f.write(f'LOSS: {epoch_loss}\n')

    print('TRAINING FINISHED')
    f.write('TRAINING FINISHED') 
    f.close()
    torch.save(model.state_dict(),'./train0/weights/multitask.pt')

if __name__ == '__main__':
    train(10)
