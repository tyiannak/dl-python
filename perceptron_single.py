import numpy as np
import plotly.graph_objects as go
import plotly

def activation_function(z):
    return 1 / (1 + np.exp(-z))

def compute_output(x, b, w_0, w_1):
    sum = np.dot(x, np.array([w_0, w_1]) + b)
    return activation_function(sum), sum

# define b, w(0), w(1):
b = 1
w_0 = 1
w_1 = -2

x = np.array([0.3, 0.6])

x_ = np.arange(-2, 2, 0.05)
y_ = np.arange(-2, 2, 0.05)
Z = np.zeros((len(y_), len(x_)))
for ix, xx in enumerate(x_):
    for iy, yy in enumerate(y_):
        Z[iy, ix], _ = compute_output(np.array([xx, yy]), b, w_0, w_1)


y, s = compute_output(x, b, w_0, w_1)
text = f'g(WX+b) = g({s:.3f}) = {y:.3f}'
fig = go.Figure(data=[
    go.Scatter(x=[x[0]], y=[x[1]], mode='markers+text', text=[text],
               textposition="top center",
               textfont=dict(
                   family="sans serif", size=28, color="orange"),
               marker=dict(size=15, color='Green',
                           line=dict(width=2, color='Orange'))),
    go.Heatmap(z=Z, y=y_, x=x_, colorscale=[[0.0, 'rgb(20, 20, 200)'], [1.0, 'rgb(200, 20, 20)']])],
    layout=go.Layout(title='x', xaxis=dict(title="x1",), yaxis=dict(title="x2",)))
plotly.offline.plot(fig)