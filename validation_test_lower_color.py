import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from PAR.convnext_extractor import ConvexNextExtractor
from PAR.resnet_extractor import ResNetExtractor
from PAR.color_classifier import ColorClassifier
from PAR.par_utils import ImageDataset
import torchvision.models as models
import torch.nn.functional as F


def calculate_accuracy(model, data_loader, extractor):
    model.eval()

    Y, Y_hat = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            Y.append(labels)
            inputs = inputs.to('cuda')
            features = extractor(inputs)
            outputs = model(features).to('cuda')
            outputs = F.softmax(outputs, dim=1)
            Y_hat.append(torch.argmax(outputs, dim=1) + 1)

    Y = torch.concatenate(Y).cpu()
    Y_hat = torch.concatenate(Y_hat).cpu()

    return (Y==Y_hat).float().mean().item()


def validate():
    model = ColorClassifier().to('cuda')
    model.load_state_dict(torch.load('./weights/lower_color_resnet50_test_model.pt'))
    transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()
    extractor = ResNetExtractor()
    validate_data = ImageDataset('./data/par_datasets/validation_set.txt','./data/par_datasets/validation_set',class_name='lower_color' ,transform=transform)
    validate_loader = torch.utils.data.DataLoader(validate_data,batch_size=64)
    acc = calculate_accuracy(model, validate_loader, extractor)
    print("accuracy:", acc)


# def validate():
#     model = ColorClassifier().to('cuda')
#     model.load_state_dict(torch.load('./weights/lower_color_resnet50_model.pt'))
#     transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()
#     extractor = ConvexNextExtractor()
#     validate_data = ImageDataset('./data/par_datasets/validation_set.txt','./data/par_datasets/validation_set',class_name='lower_color' ,transform=transform)
#     validate_loader = torch.utils.data.DataLoader(validate_data,batch_size=64)
#     acc = calculate_accuracy(model, validate_loader, extractor)
#     print("accuracy:", acc)


if __name__ == '__main__':
    validate()