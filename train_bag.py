import torch.nn as nn
import torch
import torchvision.models as models
from PAR.convnext_extractor import ConvexNextExtractor
from PAR.binary_classifier import BinaryClassiefier
from PAR.par_utils import ImageDataset


def train_one_epoch(train_loader,optimizer,model,loss_fn,extractor):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda').unsqueeze(1)
        optimizer.zero_grad()
        features = extractor(inputs)
        outputs = model(features)
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
    extractor = ConvexNextExtractor()
    model = BinaryClassiefier().to('cuda')
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),lr = LR)
    transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()
    train_data = ImageDataset('./data/par_datasets/training_set.txt','./data/par_datasets/training_set',class_name='bag' ,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64)
    model.train(True)
    print('START TRAINING')
    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}')
        epoch_loss = train_one_epoch(train_loader,optimizer,model,criterion,extractor)
        print(f'LOSS: {epoch_loss}')
    print('TRAINING FINISHED')
    torch.save(model.state_dict(),'./weights/bag_model.pt')

if __name__ == '__main__':
    train(13)
