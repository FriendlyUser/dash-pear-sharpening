import dash
import dash_html_components as html
import dash_core_components as dcc
import base64

app = dash.Dash()

list_of_images = [
    'pear.png',
    'sharpened.png'
]

app.layout = html.Div([
    dcc.Dropdown(
        id='image-dropdown',
        options=[{'label': i, 'value': i} for i in list_of_images],
        # initially display the first entry in the list
        value=list_of_images[0]
    ),
    html.Img(id='image')
])


@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('image-dropdown', 'value')])
def update_image_src(image_path):
    # print the image_path to confirm the selection is as expected
    print('current image_path = {}'.format(image_path))
    encoded_image = base64.b64encode(open(image_path, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


if __name__ == '__main__':
    app.run_server(debug=True, port=8053, host='0.0.0.0')