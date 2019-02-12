import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
from dash.dependencies import Input, Output

import solution as sol
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
image_orig = 'pear.png' # replace with your own image
image_sharp = 'sharpened.png' # replace with your own image
encoded_orig_image = base64.b64encode(open(image_orig, 'rb').read())
encoded_sharp_image = base64.b64encode(open(image_sharp, 'rb').read())
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.Div(children=[
      html.Img(
        src='data:image/png;base64,{}'.format(encoded_orig_image.decode())
      ),
    ]),
    html.Div(children=[
       html.Img(
        src='data:image/png;base64,{}'.format(encoded_sharp_image.decode())
      ),
      dcc.Slider(
        id='slider-enhancement',
        min=-5,
        max=10,
        step=0.5,
        value=-3,
      ),
    ]),
    html.Div(id='my-div'),
    html.Div(
      id='div-enhancement-factor',
      style={
          'display': 'none',
          'margin': '25px 5px 30px 0px'
    },
      children=[
          "Enhancement Factor:",
          html.Div(
              style={'margin-left': '5px'},
              children=dcc.Slider(
                  id='slider-enhancement-factor',
                  min=0,
                  max=2,
                  step=0.1,
                  value=1,
                  updatemode='drag'
              )
          )
      ]
    )
])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='slider-enhancement', component_property='value')]
)
def update_output_div(input_value):
    test_image = sol.read_image('pear.png')
    test = sol.imgfilter2d(test_image)
    print(test)
    return 'You\'ve entered "{}"'.format(input_value)
    
if __name__ == '__main__':
    app.run_server(debug=True)