import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from PAR.convnext_extractor import ConvexNextExtractor
from PAR.multi_task import DMTPAR
from PAR.par_utils import CLAHEImageDataset
import torchvision.models as models
import torch.nn.functional as F


def calculate_accuracy(model, data_loader):
    model.eval()

    Y_bag, Y_hat_bag = [], []
    Y_hat, Y_hat_hat = [], []
    Y_gender, Y_hat_gender = [], []
    Y_lower_color, Y_hat_lower_color = [], []
    Y_upper_color, Y_hat_upper_color = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to('cuda')
            upper_color, lower_color, bag_presence, hat_presence, gender = model(inputs)
            for j in range(len(labels[0, :])):
                if j == 0:
                    Y_upper_color.append(labels[:, j])
                    
                    outputs = F.softmax(upper_color, dim=1)
                    Y_hat_upper_color.append(torch.argmax(outputs, dim=1))
                elif j == 1:
                    Y_lower_color.append(labels[:, j])
                    outputs = F.softmax(lower_color, dim=1)
                    Y_hat_lower_color.append(torch.argmax(outputs, dim=1))
                elif j == 2:
                    Y_bag.append(labels[:, j])
                    Y_hat_bag.append(torch.where(bag_presence > .5, 1.0, 0.0).int())
                elif j == 3:
                    Y_hat.append(labels[:, j])
                    Y_hat_hat.append(torch.where(hat_presence > .5, 1.0, 0.0).int())
                elif j == 4:
                    Y_gender.append(labels[:, j])
                    Y_hat_gender.append(torch.where(gender > .5, 1.0, 0.0).int())
            # Y.append(labels)
            # inputs = inputs.to('cuda')
            # features = extractor(inputs)
            # outputs = model(features).to('cuda')
            # outputs = F.softmax(outputs, dim=1)
            # Y_hat.append(torch.argmax(outputs, dim=1) + 1)

    Y_bag = torch.concatenate(Y_bag).cpu()
    Y_hat_bag = torch.concatenate(Y_hat_bag).cpu()
    Y_hat = torch.concatenate(Y_hat).cpu()
    Y_hat_hat = torch.concatenate(Y_hat_hat).cpu()
    Y_gender = torch.concatenate(Y_gender).cpu()
    Y_hat_gender = torch.concatenate(Y_hat_gender).cpu()
    Y_lower_color = torch.concatenate(Y_lower_color).cpu()
    Y_hat_lower_color = torch.concatenate(Y_hat_lower_color).cpu()
    Y_upper_color = torch.concatenate(Y_upper_color).cpu()
    Y_hat_upper_color = torch.concatenate(Y_hat_upper_color).cpu()

    acc_upper_color = (Y_upper_color==Y_hat_upper_color).float().mean().item()
    acc_lower_color = (Y_lower_color==Y_hat_lower_color).float().mean().item()
    acc_bag = (Y_bag==Y_hat_bag).float().mean().item()
    acc_hat = (Y_hat==Y_hat_hat).float().mean().item()
    acc_gender = (Y_gender==Y_hat_gender).float().mean().item()

    return acc_upper_color, acc_lower_color, acc_bag, acc_hat, acc_gender





def validate():
    model = DMTPAR().to('cuda')
    model.load_state_dict(torch.load('./weights/multitask_general_model_with_clahe_test3.pt'))
    model.eval()

    transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms(antialias=True)
    validate_data = CLAHEImageDataset('./data/par_datasets/validation_set.txt','./data/par_datasets/validation_set/',transform=transform)
    validate_loader = torch.utils.data.DataLoader(validate_data,batch_size=64)
    acc_upper_color, acc_lower_color, acc_bag, acc_hat, acc_gender = calculate_accuracy(model, validate_loader)
    print("accuracy upper color:", acc_upper_color)
    print("accuracy lower color:", acc_lower_color)
    print("accuracy bag:", acc_bag)
    print("accuracy hat:", acc_hat)
    print("accuracy gender:", acc_gender)


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