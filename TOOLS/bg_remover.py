import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import cv2
import numpy as np
from TOOLS.image_transformations import ImageTransformations
# I also needed to install: pip install --upgrade typing-extensions


class BgRemover(): 

    _GREEN = [0, 1, 0]
    _BLACK = [0, 0, 0]
    _WHITE = [1, 1, 1]
    
    def __init__(self): 
        # Loads the model 
        #model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        #model.eval()
        #self.model = model
        # simple transform that makes an image to tensor 
        self.simple_transform = T.Compose([T.ToTensor()])
        # transform that normalizes the tensor so it's ready for the model 
        # self.normalization_transform = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.normalization_transform = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.imTrans = ImageTransformations(True, read_image_from_file=False)

    
    def _img2tensor(self, img_path, normalization=True):
        img = Image.open(img_path).convert("RGB")
        if normalization == True: 
            tensor = self.simple_transform(img)
            tensor = self.normalization_transform(tensor).unsqueeze(0)
        else: 
            tensor = self.simple_transform(img).unsqueeze(0)
        return tensor 
    
    def _tensor2img(self, tensor, output_img_path=None):
        '''Converts the input tensor to an image. 
        It returns the image and also saves it into output_img_path if required'''
        img = T.ToPILImage()(tensor.squeeze(0))
        if output_img_path is not None: 
            img.save(output_img_path)
        return img
    
    
    def _getMask(self, tensor):
        with torch.no_grad():
            output = self.model(tensor)['out'][0]
        output_predictions = output.argmax(0)
        # Create a mask where background pixels are set to 0
        mask = (output_predictions == 0).bool()
        return mask 
    
    def _bgRemove(self, tensor, mask, replacement_colour=[0, 1, 0]): 
        '''!!! please note !!! tensor is the NOT NORMALIZED tensor from the image'''
        res = tensor 
        res[:, 0, mask] = replacement_colour[0]
        res[:, 1, mask] = replacement_colour[1]
        res[:, 2, mask] = replacement_colour[0]
        return res
    
    def bgr_img(self, img_path, out_path, colour=[0, 1, 0]):
        '''removes the bg from an image and returns an image'''
        normalized_tensor = self._img2tensor(img_path, normalization=True)
        mask = self._getMask(normalized_tensor)
        tensor = self._img2tensor(img_path, normalization=False)
        out = self._bgRemove(tensor, mask, colour)
        out_img = self._tensor2img(out, out_path)
        return out_img
    
    def bgr_tensor(self, in_tensor, colour=[0, 1, 0]):
        '''removes the bg from an image and returns an image'''
        normalized_tensor = self.normalization_transform(in_tensor.squeeze(0)).unsqueeze(0)
        mask = self._getMask(normalized_tensor)
        out = self._bgRemove(in_tensor, mask, colour)
        return out
    

    def clahe(self, tensor):
        # prepares the tensor for opencv
        img = self._tensor2img(tensor)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # applies clahe 
        img = self.imTrans.set_image(img)
        img = self.imTrans.clahe_v2()
        # transforms the image back to a tensor of the correct type 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img)
        tensor = tensor.permute(2, 0, 1)
        return tensor


    # deprecato da me, non lo usate 
    def equalize(self, image_tensor):
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_tensor = (image_tensor[0, :] * 255).byte()
        # image_tensor = T.functional.to_tensor(image_tensor)
        # transform = T.ColorJitter(saturation=4)  # You can adjust the saturation factor
        # image_tensor = transform(image_tensor)
        equalized_image = T.functional.equalize(image_tensor)
        return equalized_image.unsqueeze(0)

    

# ----------- USAGE ---------------

# def main(): 
#     img_path = 'test.png'
#     out_path = 'output.jpg'
#     out_tensor_path = 'out_tensor.jpg'

#     bgr = bgRemover()
#     # from image ------------
#     bgr.bgr_img(img_path, out_path, bgr._WHITE)


#     # from tensor -----------
#     tensor = bgr._img2tensor(img_path, normalization=False) # creates a tensr for test 

#     tensor = bgr.clahe(tensor)  # clahe
#     tensor = bgr.bgr_tensor(tensor, bgr._BLACK)  # background removal 

#     bgr._tensor2img(tensor, out_tensor_path)
    
    

    


# if __name__ == '__main__':
#     main()



