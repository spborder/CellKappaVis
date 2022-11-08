"""

Script to convert Sedeen annotations to image

Sedeen ".session" files follow XML formatting

Key attributes:
    - sesssion version
        - image identifier
            - dimensions
            - overlays
                - graphic description --> name= "Region {N}" (starts from 0)
                    - pen style --> color="#ff0000ff" (color is in hex code)

"""

import os
import sys
import numpy as np

import lxml.etree as ET

from PIL import Image
from glob import glob

from skimage.draw import polygon
import shapely
from shapely.geometry import Polygon, Point


# Defining class object for a given set of annotations
class SedeenImage:
    def __init__(self,
                img_path: str,
                session_path: str):
    
        self.img_path = img_path
        self.session_path = session_path

        self.image_name = self.img_path.split(os.sep)[-1]

        self.annotated_img, self.annotations = self.parse_session()

    def parse_session(self):
        
        # Reading session xml file 
        sess_xml = self.read_xml(self.session_path)

        # Base is image identifier
        reg_in_img = sess_xml.getroot().findall('./image/overlays/graphic')

        img_dimensions = sess_xml.getroot().find('./image/dimensions')
        img_dimensions = [int(i) for i in img_dimensions.text.split(',')]
        img_width, img_height = img_dimensions[0], img_dimensions[1]
        
        # Iterating through regions in the image and adding them to mask
        ann_mask = np.stack((np.zeros((img_height,img_width)),np.zeros((img_height,img_width)),np.zeros((img_height,img_width))),axis=2)
        for reg in reg_in_img:
            
            # Getting the color of the region
            reg_red, reg_green, reg_blue, alpha = self.hex_to_rgb(reg.find('./pen').attrib['color'])
            # Getting the list of points (for point annotations this should have a length of 1)
            point_list = reg.findall('./point-list/point')
            
            # For multiple types of annotations, one being just points and the other being actual shapes
            if len(point_list)>0:
                ann_reg = []
                for p in point_list:
                    point_poly = Point(int(float(p.text.split(',')[1])),int(float(p.text.split(',')[0])))
                    point_poly = point_poly.buffer(5)
                    ann_reg.append(point_poly)

                # making polygon mask
                reg_mask_3d = self.make_mask(ann_reg,[img_width,img_height],[reg_red,reg_green,reg_blue])

                ann_mask+=reg_mask_3d
        # Processing combined annotations to set black background to transparent
        zero_mask = np.where(np.sum(ann_mask.copy(),axis=2)==0,0,255)
        ann_mask_4d = np.concatenate((ann_mask,zero_mask[:,:,None]),axis=-1)
        rgba_mask = Image.fromarray(np.uint8(ann_mask_4d),'RGBA')
        
        og_img = Image.open(self.img_path).convert('RGBA')
        
        og_img.paste(rgba_mask, mask = rgba_mask)

        return og_img,rgba_mask
    
    def make_mask(self,polys,img_size,color_list):

        reg_mask = np.zeros((img_size[1],img_size[0]))
        for p in polys:
            x_coords = [int(i[0]) for i in list(p.exterior.coords)]
            y_coords = [int(i[1]) for i in list(p.exterior.coords)]
            rr,cc = polygon(y_coords,x_coords,(img_size[1],img_size[0]))
            reg_mask[cc,rr] = 1
            reg_mask_3d = np.stack((reg_mask,reg_mask,reg_mask),axis=2)
            reg_mask_3d[:,:,0]*=color_list[0]
            reg_mask_3d[:,:,1]*=color_list[1]
            reg_mask_3d[:,:,2]*=color_list[2]

        return reg_mask_3d


    def hex_to_rgb(self,hex_code):
        # From: https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
        value = hex_code.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i+lv//3],16) for i in range(0,lv,lv//3))

    def read_xml(self,xml_path):
        tree = ET.parse(xml_path)
        return tree


def main():

    base_dir = 'C:\\Users\\Sam\\Desktop\\GlomAnnotationsAndCode\\OriginalAnnotations\\JET annotations-Control\\Wk15\\'
    img_dir = base_dir
    session_dir = base_dir+'sedeen/'
    save_dir = base_dir+'Processed_Images/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img in glob(img_dir+'*.tif'):

        image_name = img.split(os.sep)[-1]
        session_path = session_dir+image_name.replace('.tif','.session.xml')

        processed_img = SedeenImage(img,session_path)
        processed_img.annotated_img.save(save_dir+image_name)

if __name__=='__main__':
    main()





