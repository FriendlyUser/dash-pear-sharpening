import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
from dash.dependencies import Input, Output

import solution as sol
# Create temporary file and then delete it
from os import remove
# Get time 
from time import gmtime, strftime
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
image_orig = 'pear.png' # replace with your own image
image_sharp = 'sharpened.png' # replace with your own image
encoded_orig_image = base64.b64encode(open(image_orig, 'rb').read())
encoded_sharp_image = base64.b64encode(open(image_sharp, 'rb').read())
app.layout = html.Div(children=[
    html.H1(children='Pear Sharpening Dashboard'),
    html.Div(children='''
        ECE471: In my computer vision class I created a simple filter to sharpen a pear image and I had the great image of making an interactive dash application for it. This filter uses the approximate laplacian of gaussian for the sharpening filter.
    '''),
    html.Div(children=[
      html.H6('Original RGB Image'),
      html.Img(
        src='data:image/png;base64,{}'.format(encoded_orig_image.decode())
      ),
    ]),
    html.Div(children=[
      html.H6('Final Sharpened GrayScale Image'),
      html.Img(
        src='data:image/png;base64,{}'.format(encoded_sharp_image.decode())
      )
    ]),
    html.H6('Experiment with the Sigma And Impulse Magnitude Parameters'),
    html.Div([
      html.Div([
        html.P('Sigma Blur'),
      ], className="three columns"),
      html.Div([
        dcc.Slider(
          id='sigma-blur',
          min=0,
          max=200,
          step=0.5,
          marks={10*i: 10 * i for i in range(20)},
          value=10,
        ),
      ],className="nine columns"),
    ],className="row"),
    html.Br(),
    html.Div([
      html.Div([
        html.P('Impulse Magnitude'),
      ], className="three columns"),
      html.Div([
        dcc.Slider(
          id='impulse-magnitude',
          min=0,
          max=20,
          step=0.25,
          marks={2* i: 2 * i for i in range(10)},
          value=2
        ),
      ],className="nine columns"),
    ],className="row"),
    html.Br(),
    html.Img(id='image'),
    html.Br(),
    dcc.Markdown('''
Theory for HoG from: [learnopencv](https://www.learnopencv.com/histogram-of-oriented-gradients/)            
**Created by:** David Li, [Personal Website](http://url.surge.sh/1)
    ''')
    
])

@app.callback(
    Output('image', 'src'),
    [Input(component_id='impulse-magnitude', component_property='value'),
    Input(component_id='sigma-blur',component_property='value')]
)
def update_output_div(impulse_mag,sigma_blur):
    # Read Input Image, don't hardcode in the future
    input_image = sol.read_image('pear.png')
    # Get timestamped based file name
    curr_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    image_name = '%s%s' % (curr_time, '.png')
    print('Creating Sharpened Image %s' % image_name)
    # Applying filter from opencv
    sharpened_image = sol.imgfilter2d(input_image,sigma_blur,impulse_mag)
    sol.save_image(sharpened_image,image_name)
    # Reading image from file path
    encoded_image = base64.b64encode(open(image_name, 'rb').read())
    print('Deleting %s' % image_name)
    remove(image_name)
    return 'data:image/png;base64,{}'.format(encoded_image.decode())
    
if __name__ == '__main__':
    app.run_server(debug=True)