import torch.nn as nn
import torch
import torchvision.models as models
from PAR.multi_task import MTLoss,MTPAR
from PAR.par_utils import MTImageDataset


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
    f = open('./raports/train_bag_raport.txt','w+')
    criterion = MTLoss()
    model = MTPAR()
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),lr = LR)
    transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()
    train_data = MTImageDataset('./data/par_datasets/training_set.txt','./data/par_datasets/training_set' ,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64)
    model.train(True)
    print('START TRAINING')
    f.write('START TRAINING\n')
    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}')
        f.write(f'EPOCH {epoch + 1}\n')
        epoch_loss = train_one_epoch(train_loader,optimizer,model,criterion)
        print(f'LOSS: {epoch_loss}')
        f.write(f'LOSS: {epoch_loss}\n')
    print('TRAINING FINISHED')
    f.write('TRAINING FINISHED')
    f.close()
    torch.save(model.state_dict(),'./weights/multi_model.pt')

if __name__ == '__main__':
    train(13)