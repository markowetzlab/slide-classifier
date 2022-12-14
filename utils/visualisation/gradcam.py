"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
import argparse
import os
import pickle

from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate grad-cam images.')
    parser.add_argument("-stain", required=True, help="he or p53")
    parser.add_argument("-model", default='models_to_gradcam/partSize-p53-80-20-vgg19_bn_14-7-2022-epoch-6_ft.pt', help="which model to infer gradcam on")  #partSize-he-106-27-vgg19_bn_19-5-2022-epoch-39_ft.pt'
    parser.add_argument("-tilefolder", default='/media/berman01/Seagate Ext/p53_tiles_12_july_2022', help="path to folder containing case folders containing class folders containing tiles")
    parser.add_argument("-whichcase", default='models_to_gradcam/p53_trainval_split.p', help="which case within tilefolder to generate gradcams for; if a trainval_split.p file is included, will be run on all validation cases")
    parser.add_argument("-outputfolder", required=True, help="where to output tiles")
    parser.add_argument("-whichclasses", default=None, help="which classes to extract gradcams for, where classes are separated by commas; default is all classes")
    args = parser.parse_args()

    if args.stain == 'he':
        classes = ['artifact', 'atypia', 'background', 'gastric_cardia', 'immune_cells', 'intestinal_metaplasia', 'respiratory_mucosa', 'squamous_mucosa']
    elif args.stain == 'p53':
        classes = ['aberrant_positive_columnar', 'artifact', 'background', 'immune_cells', 'squamous_mucosa', 'wild_type_columnar']
    else:
        raise Warning('-stain must be he or p53')

    if args.whichclasses:
        which_classes = args.whichclasses.split(',')
    else:
        which_classes = classes

    if args.whichcase[-2:] == '.p':
        trainval_split = pickle.load(open(args.whichcase, 'rb'))
        val_cases = []
        for val_case in trainval_split['val_cases']:
            val_cases.append(os.path.split(val_case)[-1].replace('.ndpi', ''))
        val_cases.sort()
    else:
        val_cases = [args.whichcase]#os.path.split(args.tilefolder)[-1]]
    print('val_cases:', val_cases)

    for val_case in val_cases:
        print('val case:', val_case)
        for class_index, class_name in enumerate(classes):

            if class_name not in which_classes:
                continue

            os.makedirs(os.path.join(args.outputfolder, val_case, class_name), exist_ok=True)

            tile_paths = [tile for tile in os.listdir(os.path.join(args.tilefolder, val_case, class_name)) if os.path.isfile(os.path.join(args.tilefolder, val_case, class_name, tile))]
            print('class_name:', class_name)
            print('class_index:', class_index)
            print('tile_paths:', tile_paths)

            # Get params
            #target_example = 11  # Snake
            for tile_path in tile_paths:
                (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
                    get_example_params(os.path.join(args.tilefolder, val_case, class_name, tile_path), class_index, len(classes), model_path=args.model) #'models_to_gradcam/partSize-he-106-27-alexnet_19-5-2022-epoch-39_ft.pt')
                # Grad cam
                grad_cam = GradCam(pretrained_model, target_layer=51)#11)
                # Generate cam mask
                cam = grad_cam.generate_cam(prep_img, target_class)
                # Save mask
                save_class_activation_images(original_image, cam, file_name_to_export, os.path.join(args.outputfolder, val_case, class_name), alpha_value=0.6)


    print('Grad cam completed')
