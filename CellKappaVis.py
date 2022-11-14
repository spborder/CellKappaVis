"""

Codes to calculate Cohen's Kappa scores between different combinations of annotators

Some basic visualization implementation


"""

import os
import sys
import numpy as np
import pandas as pd

from glob import glob

from PIL import Image
import cv2

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import dcc, ctx

from skimage.measure import label,regionprops_table
from scipy.spatial import cKDTree
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from dash_extensions.enrich import DashProxy, html, Input, Output, MultiplexerTransform

# Reading in images and separating them by rater
image_dir ='/mnt/c/Users/Sam/Desktop/GlomAnnotationsAndCode/GlomAnnotationsAndCode/'
output_dir = '/mnt/c/Users/Sam/Desktop/GlomAnnotationsAndCode/Outputs'
annotators = os.listdir(image_dir)

"""
annotation_codes = {
    'Mesangial':{'light':(150,200,200),'dark':(180,255,255)},
    'Endothelial':{'light':(30,100,0),'dark':(70,255,255)},
    'Podocyte':{'light':(80,175,200),'dark':(120,180,215)},
}
"""
annotation_codes = {
    'Mesangial':[255,0,0],
    'Endothelial':[0,255,0],
    'Podocyte':[0,0,255]
}

annotator_img_dict = {}
for a in annotators:
    annotator_img_dict[a] = {}

    ann_img_dir = image_dir+a+'/*.tif'
    imgs_annotated = []
    for img in glob(ann_img_dir):
        img_name = img.split(os.sep)[-1]
        annotator_img_dict[a][img_name] = {}
        annotator_img_dict[a][img_name]['Image'] = np.array(Image.open(img))[:,:,0:3]
        imgs_annotated.append(img_name)

        # Generating segmentations for each image
        #hsv_img = cv2.cvtColor(annotator_img_dict[a][img_name]['Image'],cv2.COLOR_RGB2HSV)
        for cell in annotation_codes:
            #thresh_img = hsv_img.copy()

            #lower_bounds = annotation_codes[cell]['light']
            #upper_bounds = annotation_codes[cell]['dark']
            #cell_mask = cv2.inRange(thresh_img,lower_bounds,upper_bounds)
            cell_color = annotation_codes[cell]
            thresh_img = annotator_img_dict[a][img_name]['Image'].copy()
            red_mask = thresh_img[:,:,0]==cell_color[0]
            green_mask = thresh_img[:,:,1]==cell_color[1]
            blue_mask = thresh_img[:,:,2]==cell_color[2]
            cell_mask = red_mask&green_mask&blue_mask

            # Show segmented dots for each cell type
            #fig = px.imshow(Image.fromarray(np.uint8(255*cell_mask)))
            #fig.show()
            object_centroids = pd.DataFrame(regionprops_table(label(cell_mask),properties=['centroid','area']))
            object_centroids = object_centroids[object_centroids['area']>1]
            object_centroids['Cell_Type'] = [cell]*object_centroids.shape[0]
            annotator_img_dict[a][img_name][cell] = object_centroids

        annotator_img_dict[a][img_name]['All_Cell_Types'] = pd.concat([annotator_img_dict[a][img_name][i] for i in annotation_codes],ignore_index=True)

imgs_annotated = np.unique(imgs_annotated).tolist()
initial_img = annotator_img_dict[annotators[0]][imgs_annotated[0]]['Image']

initial_figure = make_subplots(rows=1,cols=2,subplot_titles = (f'Annotator 1: {annotators[0]}',f'Annotator 2: {annotators[0]}'))
imgs_to_include = [
                    px.imshow(initial_img),
                    px.imshow(initial_img)
                    ]

for i, figure in enumerate(imgs_to_include):
    if 'data' in figure:
        for trace in range(len(figure['data'])):
            initial_figure.append_trace(figure['data'][trace],row=1,col=i+1)
    else:
        initial_figure.append_trace(figure,row=1,col=i+1)

main_layout = html.Div([
    # Header
    html.Div([
        html.H1('Analyzing agreement between multiple annotators')
    ]),

    # Where images will be displayed
    html.Div([
        html.Div([
            dcc.Graph(
                id = 'compare-images',
                figure = initial_figure
            )
        ]),
        html.Div([
            dcc.Dropdown(
                imgs_annotated+['All'],
                imgs_annotated[0],
                id = 'image-name'
            ),
            html.Br(),
            dcc.Dropdown(
                annotators+['All'],
                annotators[0],
                id = 'rater-1'
            ),
            html.Br(),
            dcc.Dropdown(
                annotators+['All'],
                annotators[0],
                id = 'rater-2'
            ),
            html.Br(),
            dcc.RadioItems(
                ['Include Unscored','Ignore Unscored'],
                'Include Unscored',
                id = 'inc-unscored'
            )
        ])
    ]),

    # Where confusion matrix(ces) will be displayed
    html.Div([
        dcc.Graph(
            id = 'confusion-mat',
            figure = go.Figure()
        )
    ])

])


class CellKappaVis:
    def __init__(self,app,layout,img_dir,annotation_codes, annotator_img_dict):

        self.app = app
        self.app.layout = layout
        self.img_dir = img_dir
        self.annotation_codes = annotation_codes
        self.annotator_img_dict = annotator_img_dict

        self.include_unscored = True

        self.app.callback(
            [Output('compare-images','figure'),Output('confusion-mat','figure')],
            [Input('rater-1','value'),Input('rater-2','value'),Input('image-name','value'),Input('inc-unscored','value')]
        )(self.update_raters_image)

    def update_raters_image(self,rater_1,rater_2,img_name,include_unscored):
        
        if include_unscored=='Include Unscored':
            self.include_unscored = True
        else:
            self.include_unscored = False

        if not 'All' in [rater_1,rater_2]:
            # Comparing two specific raters
            new_conf_mat, new_kappa = self.generate_new_confusion_matrix(rater_1,rater_2,img_name)

            if not img_name=='All':
                # One specific image from both raters
                if img_name in self.annotator_img_dict[rater_1] and img_name in self.annotator_img_dict[rater_2]:
                    image_1 = self.annotator_img_dict[rater_1][img_name]['Image']
                    image_2 = self.annotator_img_dict[rater_2][img_name]['Image']

                    new_img_figure = self.make_img_subplots([image_1,image_2], [rater_1, rater_2])
                else:
                    try:
                        image = self.annotator_img_dict[rater_1][img_name]['Image']
                        new_img_figure = go.Figure(px.imshow(image).data[0])
                        new_img_figure.update_layout(title=f'Only annotated by {rater_1}')
                    except:
                        try:
                            image = self.annotator_img_dict[rater_2][img_name]['Image']
                            new_img_figure = go.Figure(px.imshow(image).data[0])
                            new_img_figure.update_layout(title=f'Only annotated by {rater_2}')   
                        except:
                            new_img_figure = go.Figure(px.imshow(np.zeros((200,200))))
                            new_img_figure.update_layout(title = 'Not annotated by either rater')
            else:
                # All images from both raters
                overlap_image_list = list(set(list(self.annotator_img_dict[rater_1].keys())) & set(list(self.annotator_img_dict[rater_2].keys())))

                image_1 = self.annotator_img_dict[rater_1][overlap_image_list[0]]['Image']
                image_2 = self.annotator_img_dict[rater_2][overlap_image_list[0]]['Image']

                new_img_figure = self.make_img_subplots([image_1,image_2],[rater_1,rater_2])

                conf_mat_list = []
                kappa_list = []
                for image in overlap_image_list:
                    new_conf_mat, new_kappa = self.generate_new_confusion_matrix(rater_1,rater_2,image)
                    conf_mat_list.append(new_conf_mat)
                    kappa_list.append(new_kappa)

                new_conf_mat = np.sum(np.array(conf_mat_list),axis=0)
                new_kappa = np.mean(kappa_list)

        else:
            # Comparing all raters to one or all raters
            if rater_1=='All' and not rater_2 == 'All':
                # Comparing one rater to all
                raters_list_1 = [i for i in self.annotator_img_dict if not i == rater_2]
                raters_list_2 = [rater_2]
            elif not rater_1=='All' and rater_2=='All':
                # Comparing one rater to all
                raters_list_1 = [rater_1]
                raters_list_2 = [i for i in self.annotator_img_dict if not i == rater_1]
            else:
                # Comparing all to all
                raters_list_1 = [i for i in self.annotator_img_dict]
                raters_list_2 = [i for i in self.annotator_img_dict]

            # Show images for every rater no matter what but show a specific image for each rater if All is not 
            # selected for image
            if not img_name == 'All':
                image_list = [self.annotator_img_dict[r][img_name]['Image'] if img_name in self.annotator_img_dict[r] else np.zeros((200,200)) for r in self.annotator_img_dict]

                new_img_figure = self.make_img_subplots(image_list,list(self.annotator_img_dict.keys()))

                conf_mat_list = []
                kappa_list = []
                for r1 in raters_list_1:
                    for r2 in raters_list_2:
                        # Not comparing a rater to themselves
                        if not r1==r2:
                            new_conf_mat, new_kappa = self.generate_new_confusion_matrix(r1,r2,img_name)
                            if not type(new_kappa)==str:
                                conf_mat_list.append(new_conf_mat)
                                kappa_list.append(new_kappa)
                
                new_conf_mat = np.sum(np.array(conf_mat_list),axis=0)
                new_kappa = np.mean(kappa_list)
            else:
                # Show example image
                images_rater_0 = list(self.annotator_img_dict[list(self.annotator_img_dict.keys())[0]].keys())[0]
                image_list = [self.annotator_img_dict[r][images_rater_0]['Image'] if images_rater_0 in self.annotator_img_dict[r] else np.zeros((200,200)) for r in self.annotator_img_dict]

                new_img_figure = self.make_img_subplots(image_list,list(self.annotator_img_dict.keys()))

                conf_mat_list = []
                kappa_list = []
                for r1 in raters_list_1:
                    for r2 in raters_list_2:
                        # Not comparing a rater to themselves
                        if not r1==r2:
                            # Finding overlapping images between two raters
                            imgs_r1 = list(self.annotator_img_dict[r1].keys())
                            imgs_r2 = list(self.annotator_img_dict[r2].keys())

                            overlap_list = np.unique(list(set(imgs_r1)&set(imgs_r2)))
                            for img in overlap_list:
                                new_conf_mat, new_kappa = self.generate_new_confusion_matrix(r1,r2,img)
                                conf_mat_list.append(new_conf_mat)
                                kappa_list.append(new_kappa)
                
                new_conf_mat = np.sum(np.array(conf_mat_list),axis=0)
                new_kappa = np.mean(kappa_list)
        
        x_axes_labels = [rater_1+'_'+i for i in self.annotation_codes]
        y_axes_labels = [rater_2+'_'+i for i in self.annotation_codes]

        if np.shape(new_conf_mat)[0]>len(x_axes_labels):
            x_axes_labels+=[rater_1+'_Unscored']
            y_axes_labels+=[rater_2+'_Unscored']

        x_axes_labels.reverse()

        # Constructing the confusion matrix with labeled axes and title
        conf_mat_figure = go.Figure()
        conf_img_data = px.imshow(np.flip(new_conf_mat,axis=1),x = x_axes_labels, y = y_axes_labels,text_auto=True)
                        
        for trace in conf_img_data['data']:
            conf_mat_figure.add_trace(trace)

        if not type(new_kappa) == str:
            new_kappa = round(new_kappa,3)

        conf_mat_figure.update_layout(title=f'Kappa Score between {rater_1} and {rater_2} = {new_kappa}')

        return new_img_figure, conf_mat_figure

    def make_img_subplots(self,image_list, rater_list):
        new_figure = make_subplots(rows=1,cols=len(image_list),subplot_titles = tuple([f'Annotator {i}: {rater_list[i]}' for i in range(len(rater_list))]))
        imgs_to_include = [px.imshow(i) for i in image_list]

        for i, figure in enumerate(imgs_to_include):
            if 'data' in figure:
                for trace in range(len(figure['data'])):
                    new_figure.append_trace(figure['data'][trace],row=1,col=i+1)
            else:
                new_figure.append_trace(figure,row=1,col=i+1)

        return new_figure

    def generate_new_confusion_matrix(self, rater_1, rater_2, image_name):
        
        if image_name not in self.annotator_img_dict[rater_1] or image_name not in self.annotator_img_dict[rater_2]:
            conf_mat = np.zeros((3,3))
            kappa_score = f'{image_name} not annotated by both'
        else:
            # In this case, the centroids are stored in a dict format so 
            centroids_1 = self.annotator_img_dict[rater_1][image_name]['All_Cell_Types']
            centroids_2 = self.annotator_img_dict[rater_2][image_name]['All_Cell_Types']

            # KDTree search both sets of centroids for nearest points in other set of labels
            tree_1 = cKDTree(centroids_1.values[:,0:2])
            distances1, idxes1 = tree_1.query(centroids_2.values[:,0:2])
            include_idxes_1 = np.unique(idxes1[distances1<=15])

            tree_2 = cKDTree(centroids_2.values[:,0:2])
            distances2, idxes2 = tree_2.query(centroids_1.values[:,0:2])
            include_idxes_2 = np.unique(idxes2[distances2<=15])

            # Now we have overlapping indices for both rater_1 and rater_2
            rater_1_overlap = pd.DataFrame({rater_1:centroids_1.iloc[include_idxes_1,-1]},index=include_idxes_1)
            rater_2_overlap = pd.DataFrame({rater_2:centroids_2.iloc[include_idxes_2,-1]},index=include_idxes_2)

            overlapping_cells = pd.concat([rater_1_overlap,rater_2_overlap],axis=1)

            if not self.include_unscored:
                overlapping_cells = overlapping_cells.dropna()
            else:
                overlapping_cells = overlapping_cells.fillna('Unscored')

            # hacky way to output overlap dataframes
            #output_dir = '/mnt/c/Users/Sam/Desktop/GlomAnnotationsAndCode/Outputs/'
            #if not rater_1==rater_2:
            #    overlapping_cells.to_csv(output_dir+f'{rater_1}_{rater_2}_{image_name.replace(".tif",".csv")}')

            conf_mat = confusion_matrix(overlapping_cells.iloc[:,0],overlapping_cells.iloc[:,1])
            kappa_score = cohen_kappa_score(overlapping_cells.iloc[:,0],overlapping_cells.iloc[:,1])

        return conf_mat, kappa_score



if __name__ == '__main__':

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    main_app = DashProxy(__name__, external_stylesheets = external_stylesheets, transforms = [MultiplexerTransform()])

    cell_kappa_vis_app = CellKappaVis(main_app, main_layout, image_dir, annotation_codes, annotator_img_dict)
    cell_kappa_vis_app.app.run_server(debug=True)


