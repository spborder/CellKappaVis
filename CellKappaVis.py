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
annotators = os.listdir(image_dir)

annotation_codes = {
    'Mesangial':{'light':(150,200,200),'dark':(180,255,255)},
    'Endothelial':{'light':(30,100,0),'dark':(70,255,255)},
    'Podocyte':{'light':(80,175,200),'dark':(120,180,215)},
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
        #hsv_img = np.array(annotator_img_dict[a][img_name]['Image'].convert('HSV'))
        hsv_img = cv2.cvtColor(annotator_img_dict[a][img_name]['Image'],cv2.COLOR_RGB2HSV)
        for cell in annotation_codes:
            thresh_img = hsv_img.copy()

            lower_bounds = annotation_codes[cell]['light']
            upper_bounds = annotation_codes[cell]['dark']
            thresh_img = cv2.inRange(thresh_img,lower_bounds, upper_bounds)

            # Show segmented dots for each cell type
            #fig = px.imshow(Image.fromarray(np.uint8(255*thresh_img)))
            #fig.show()
            object_centroids = pd.DataFrame(regionprops_table(label(thresh_img),properties=['centroid']))
            object_centroids['Cell_Type'] = [cell]*object_centroids.shape[0]
            annotator_img_dict[a][img_name][cell] = object_centroids

        annotator_img_dict[a][img_name]['All_Cell_Types'] = pd.concat([annotator_img_dict[a][img_name][i] for i in annotation_codes],ignore_index=True)

imgs_annotated = np.unique(imgs_annotated)
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


# In this case, the centroids are stored in a dict format so 
centroids_1 = annotator_img_dict[annotators[0]][imgs_annotated[0]]['All_Cell_Types']
centroids_2 = annotator_img_dict[annotators[0]][imgs_annotated[0]]['All_Cell_Types']

# KDTree search
tree = cKDTree(centroids_1.values[:,0:2])
distances, idxes = tree.query(centroids_2.values[:,0:2])


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
                imgs_annotated,
                imgs_annotated[0],
                id = 'image-name'
            ),
            html.Br(),
            dcc.Dropdown(
                annotators,
                annotators[0],
                id = 'rater-1'
            ),
            html.Br(),
            dcc.Dropdown(
                annotators,
                annotators[0],
                id = 'rater-2'
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

        self.app.callback(
            [Output('compare-images','figure'),Output('confusion-mat','figure')],
            [Input('rater-1','value'),Input('rater-2','value'),Input('image-name','value')]
        )(self.update_raters_image)

    def update_raters_image(self,rater_1,rater_2,img_name):
        
        # Images from both raters
        image_1 = self.annotator_img_dict[rater_1][img_name]['Image']
        image_2 = self.annotator_img_dict[rater_2][img_name]['Image']

        new_img_figure = self.make_img_subplots(image_1,image_2, rater_1, rater_2)
        new_conf_mat, new_kappa = self.generate_new_confusion_matrix(rater_1,rater_2,img_name)

        x_axes_labels = [rater_1+'_'+i for i in self.annotation_codes]
        y_axes_labels = [rater_2+'_'+i for i in self.annotation_codes]

        # Constructing the confusion matrix with labeled axes and title
        conf_mat_figure = go.Figure()
        conf_img_data = px.imshow(new_conf_mat,x = x_axes_labels, y = y_axes_labels,text_auto=True,
                        title=f'Kappa Score between {rater_1} and {rater_2} = {new_kappa}')
                        
        for trace in conf_img_data['data']:
            conf_mat_figure.add_trace(trace)
        conf_mat_figure.update_layout()

        return new_img_figure, conf_mat_figure

    def make_img_subplots(self,image_1,image_2, rater_1, rater_2):
        new_figure = make_subplots(rows=1,cols=2,subplot_titles = (f'Annotator 1: {rater_1}',f'Annotator 2: {rater_2}'))
        imgs_to_include = [
                            px.imshow(image_1),
                            px.imshow(image_2)
                            ]

        for i, figure in enumerate(imgs_to_include):
            if 'data' in figure:
                for trace in range(len(figure['data'])):
                    new_figure.append_trace(figure['data'][trace],row=1,col=i+1)
            else:
                new_figure.append_trace(figure,row=1,col=i+1)

        return new_figure

    def generate_new_confusion_matrix(self, rater_1, rater_2, image_name):
        
        # In this case, the centroids are stored in a dict format so 
        centroids_1 = self.annotator_img_dict[rater_1][image_name]['All_Cell_Types']
        centroids_2 = self.annotator_img_dict[rater_2][image_name]['All_Cell_Types']

        # KDTree search
        tree = cKDTree(centroids_1.values[:,0:2])
        distances, idxes = tree.query(centroids_2.values[:,0:2])
        include_idxes = idxes[distances<=15]
        unscored_idxes = idxes[distances>15]

        overlapping_cells = pd.DataFrame({rater_1:centroids_1.iloc[include_idxes,-1],rater_2:centroids_2.iloc[include_idxes,-1]})
        
        conf_mat = confusion_matrix(overlapping_cells[rater_1],overlapping_cells[rater_2])
        kappa_score = cohen_kappa_score(overlapping_cells[rater_1],overlapping_cells[rater_2])

        return conf_mat, kappa_score



if __name__ == '__main__':

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    main_app = DashProxy(__name__, external_stylesheets = external_stylesheets, transforms = [MultiplexerTransform()])

    cell_kappa_vis_app = CellKappaVis(main_app, main_layout, image_dir, annotation_codes, annotator_img_dict)
    cell_kappa_vis_app.app.run_server(debug=True)


