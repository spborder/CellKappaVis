"""

Converting student annotations so that colors are either Red, Green, or Blue

"""

import os
import numpy as np

from glob import glob
from PIL import Image

import cv2

# Reading in images and separating them by rater
#image_dir ='/mnt/c/Users/Sam/Desktop/GlomAnnotationsAndCode/GlomAnnotationsAndCode/'
image_dir = 'C:\\Users\\Sam\\Desktop\\GlomAnnotationsAndCode\\GlomAnnotationsAndCode\\'
annotators = os.listdir(image_dir)

annotation_codes = {
    'Mesangial':{'light':(150,200,200),'dark':(180,255,255)},
    'Endothelial':{'light':(30,100,0),'dark':(70,255,255)},
    'Podocyte':{'light':(80,175,200),'dark':(120,180,215)},
}

for a in annotators:

    if not a == 'Dr.T':

        ann_img_dir = image_dir+a+'/*.tif'
        save_img_dir = image_dir+a+'/Processed_Images/'

        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)

        imgs_annotated = []
        for img in glob(ann_img_dir):
            img_name = img.split(os.sep)[-1]
            image = np.array(Image.open(img))[:,:,0:3]

            end_image = np.zeros(np.shape(image))

            # Generating segmentations for each image
            hsv_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
            for channel,cell in enumerate(annotation_codes):
                thresh_img = hsv_img.copy()

                lower_bounds = annotation_codes[cell]['light']
                upper_bounds = annotation_codes[cell]['dark']
                thresh_img = cv2.inRange(thresh_img,lower_bounds, upper_bounds)

                end_image[:,:,channel] = np.where(thresh_img>0,255,0)
            
            # Adding alpha channel
            zero_mask = np.where(np.sum(end_image.copy(),axis=2)==0,0,255)
            end_image_4d = np.concatenate((end_image,zero_mask[:,:,None]),axis=-1)
            rgba_mask = Image.fromarray(np.uint8(end_image_4d),'RGBA')

            og_img = Image.open(img).convert('RGBA')
            og_img.paste(rgba_mask,mask=rgba_mask)

            og_img.save(save_img_dir+img_name)








