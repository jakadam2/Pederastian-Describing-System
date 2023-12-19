import torch.nn as nn
import torch
import torchvision.models as models
from PAR.resnet_extractor import Resnet50Extractor
from PAR.gender_classifier import GenderClassiefierResNet
from PAR.par_utils import ImageDataset


def train_one_epoch(train_loader,optimizer,model,loss_fn):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data
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
    criterion = nn.BCELoss()
    model = GenderClassiefierResNet(Resnet50Extractor()).to('cuda')
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),lr = LR)
    transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    train_data = ImageDataset('./data/par_datasets/training_set.txt','./data/par_datasets/training_set',class_name='gender' ,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64)
    model.train(True)
    print('START TRAINING')
    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}')
        epoch_loss = train_one_epoch(train_loader,optimizer,model,criterion)
        print(f'LOSS: {epoch_loss}')
    print('TRAINING FINISHED')
    torch.save(model.state_dict(),'resnetgender.pt')

if __name__ == '__main__':
    train(13)
