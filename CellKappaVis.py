"""

Codes to calculate Cohen's Kappa scores between different combinations of annotators

Some basic visualization implementation


"""

import os
import sys
import numpy as np

from glob import glob

from PIL import Image

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import dcc, ctx

from dash_extensions.enrich import DashProxy, html, Input, Output, MultiplexerTransform

# Reading in images and separating them by rater
image_dir ='/mnt/c/Users/Sam/Desktop/GlomAnnotationsAndCode/GlomAnnotationsAndCode/'
annotators = os.listdir(image_dir)

annotation_codes = {
    'Mesangial':[],
    'Endothelial':[],
    'Podocyte':[],
    'Unknown':[]
}

annotator_img_dict = {}
for a in annotators:
    annotator_img_dict[a] = {}

    ann_img_dir = image_dir+a+'/*.tif'
    imgs_annotated = []
    for img in glob(ann_img_dir):
        img_name = img.split(os.sep)[-1]
        annotator_img_dict[a][img_name] = Image.open(img)
        imgs_annotated.append(img_name)

imgs_annotated = np.unique(imgs_annotated)
initial_img = annotator_img_dict[annotators[0]][imgs_annotated[0]]

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
        image_1 = self.annotator_img_dict[rater_1][img_name]
        image_2 = self.annotator_img_dict[rater_2][img_name]

        new_img_figure = self.make_img_subplots(image_1,image_2, rater_1, rater_2)
        conf_mat_placeholder = go.Figure()

        return new_img_figure, conf_mat_placeholder

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



if __name__ == '__main__':

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    main_app = DashProxy(__name__, external_stylesheets = external_stylesheets, transforms = [MultiplexerTransform()])

    cell_kappa_vis_app = CellKappaVis(main_app, main_layout, image_dir, annotation_codes, annotator_img_dict)
    cell_kappa_vis_app.app.run_server(debug=True)


