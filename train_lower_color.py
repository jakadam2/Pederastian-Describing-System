import torch.nn as nn
import torch
import torchvision.models as models
from PAR.resnet_extractor import ResNetExtractor
from PAR.color_classifier import ColorClassifier
from PAR.par_utils import ImageDataset
import torch.nn.functional as F

def one_hot_encoding(labels, num_classes=11):
    "Perform one hot encoding"
    new_labels = torch.zeros(len(labels), num_classes).cuda()
    for i in range(len(labels)):
        new_labels[i, int(labels[i]) - 1] = 1
    return new_labels

def train_one_epoch(train_loader,optimizer,model,loss_fn,extractor):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')#.unsqueeze(1)
        labels = labels.long() - 1
        # inputs, labels = inputs, labels.unsqueeze(1)
        optimizer.zero_grad()
        
        features = extractor(inputs)
        
        outputs = model(features)
        #labels = one_hot_encoding(labels)

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
    f = open('./raports/train_lower_color_resnet50_raport.txt','w+')
    criterion = nn.CrossEntropyLoss()
    extractor = ResNetExtractor()
    model = ColorClassifier().to('cuda')
    transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),lr = LR)
    train_data = ImageDataset('./data/par_datasets/training_set.txt','./data/par_datasets/training_set',class_name='lower_color', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64)
    model.train(True)
    print('START TRAINING')
    f.write('START TRAINING\n')
    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}')
        f.write(f'EPOCH {epoch + 1}\n')
        
        epoch_loss = train_one_epoch(train_loader,optimizer,model,criterion,extractor)
        print(f'LOSS: {epoch_loss}')
        f.write(f'LOSS: {epoch_loss}\n')
    print('TRAINING FINISHED')
    f.write('TRAINING FINISHED')
    f.close()
    torch.save(model.state_dict(),'./weights/lower_color_resnet50_abonedij_model.pt')
    


if __name__ == '__main__':
    train(1)
