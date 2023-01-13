import torch.nn as nn
from torchvision import models

def get_network(network_name, class_names, pretrained=True): 
    params = {'network_name': str(network_name), 'classes': class_names}

    model = ['vgg11_bn', 'vgg16', 'vgg19_bn', 'resnet18', 'resnet152', 'squeezenet1_1', 'densenet121', 'densenet161', 'alexnet',
                 'resnext101_32x8d', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v3_large', 'wide_resnet101_2', 'mnasnet1_0',
                 'efficientnet_b0', 'efficientnet_b7', 'regnet_x_32gf', 'regnet_y_32gf', 'vit_b_16', 'vit_l_16', 'convnext_tiny', 'convnext_large', 'inception_v3']

    if params['network_name'] not in model:
        print(model)
        raise NotImplementedError('Please choose from the available networks above.')

    if network_name == 'inception_v3': 
        params['patch_size'] = 299
    else: 
        params['patch_size'] = 224

    batch_sizes = {"resnet18": 438, "resnet152": 50, "vgg11_bn": 94, "vgg16": 90, 
    "vgg19_bn": 52, "inception_v3": 96, "alexnet": 1988, "squeezenet1_1": 490, 
    "densenet121": 75, "densenet161": 37, "resnext101_32x8d": 32, "googlenet": 206, 
    "shufflenet_v2_x1_0": 480, "mobilenet_v3_large": 222, "wide_resnet101_2": 46, 
    "mnasnet1_0": 220, "efficientnet_b0": 11, "efficientnet_b7": 1, "regnet_x_32gf": 27, 
    "regnet_y_32gf": 27, "vit_b_16": 51, "vit_l_16": 10, "convnext_tiny": 86, "convnext_large": 17 
    }

    params['batch_size'] = batch_sizes[network_name]

    if network_name == "resnet18":
        model_ft = models.resnet18(pretrained=pretrained)
        model_ft.fc = nn.Linear(512, len(class_names))
    elif network_name == "resnet152":
        model_ft = models.resnet152(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "inception_v3":
        model_ft = models.inception_v3(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(768, len(class_names))
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif network_name == "vgg11_bn":
        model_ft = models.vgg11_bn(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif network_name == "vgg16":
        model_ft = models.vgg16(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif network_name == "vgg19_bn":
        model_ft = models.vgg19_bn(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif network_name == "densenet121":
        model_ft = models.densenet121(pretrained=pretrained)
        model_ft.classifier = nn.Linear(1024, len(class_names))
    elif network_name == "densenet161":
        model_ft = models.densenet161(pretrained=pretrained)
        model_ft.classifier = nn.Linear(2208, len(class_names))
    elif network_name == "alexnet":
        model_ft = models.alexnet(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, len(class_names))
    elif network_name == "squeezenet1_1":
        model_ft = models.squeezenet1_1(pretrained=pretrained)
        model_ft.classifier[1] = nn.Conv2d(
            512, len(class_names), kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = len(class_names)
    elif network_name == "resnext101_32x8d":
        model_ft = models.resnext101_32x8d(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "googlenet":
        model_ft = models.googlenet(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "shufflenet_v2_x1_0":
        model_ft = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "mobilenet_v3_large":
        model_ft = models.mobilenet_v3_large(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(1280, len(class_names))
    elif network_name == "wide_resnet101_2":
        model_ft = models.wide_resnet101_2(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "mnasnet1_0":
        model_ft = models.mnasnet1_0(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(1280, len(class_names))
    elif network_name == "efficientnet_b0":
        model_ft = models.efficientnet_b0(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(1280, len(class_names))
    elif network_name == "efficientnet_b7":
        model_ft = models.efficientnet_b7(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(2560, len(class_names))
    elif network_name == "regnet_x_32gf":
        model_ft = models.regnet_x_32gf(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "regnet_y_32gf":
        model_ft = models.regnet_y_32gf(pretrained=pretrained)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))
    elif network_name == "vit_b_16":
        model_ft = models.vit_b_16(pretrained=pretrained)
        model_ft.heads[-1] = nn.Linear(768, len(class_names))
    elif network_name == "vit_l_16":
        model_ft = models.vit_l_16(pretrained=pretrained)
        model_ft.heads[-1] = nn.Linear(1024, len(class_names))
    elif network_name == "convnext_tiny":
        model_ft = models.convnext_tiny(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(768, len(class_names))
    elif network_name == "convnext_large":
        model_ft = models.convnext_large(pretrained=pretrained)
        model_ft.classifier[-1] = nn.Linear(1536, len(class_names))

    return model_ft, params