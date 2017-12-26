import plotly as py
import plotly.graph_objs as go
import numpy as np
N = 500

def plot_predictions_3d(x,y):
    x1= x[:,1]
    y1= x[:,2]
    z1= y.flatten()

    trace0 = go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='markers',
        marker=dict(
            size=12,
            color=z1,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )
    data = [trace0]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),

    title = 'sigmoid(f(x,w))',
        scene=dict(
            xaxis = dict(
                title='x1 Axis',
            ),
            yaxis = dict(
                    title='x2 Axis',
            ),
            zaxis = dict(
                title='y Axis',
            )
        )

    )

    fig = dict(data=data, layout=layout)
    py.offline.plot(fig, filename='pred_3d.html')

def plot_predictions_2d(y):
    y = y.flatten()
    trace0 = go.Scatter(
        #x = x[:,1],
        y = y,
        name = 'Above',
        mode = 'markers',
        marker = dict(
            size = 10,
            color = 'rgba(152, 0, 0, .8)',
            line = dict(
                width = 2,
                color = 'rgb(0, 0, 0)'
            )
        )
    )

    data = [trace0]

    layout = dict(title = 'sigmoid(f(x,w))',
                 )

    fig = dict(data=data, layout=layout)
    py.offline.plot(fig, filename='pred_2d.html')

def plot_inp_out(x, y):
    x1= x[:,1]
    y1= x[:,2]
    z1= y

    trace0 = go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='markers',
        marker=dict(
            size=12,
            color=z1,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )
    data = [trace0]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),

    title = 'Training Data',
        scene=dict(
            xaxis = dict(
                title='x1 Axis',
            ),
            yaxis = dict(
                    title='x2 Axis',
            ),
            zaxis = dict(
                title='y Axis',
            )
        )

    )
    fig = dict(data=data, layout=layout)
    py.offline.plot(fig, filename='in_out.html')