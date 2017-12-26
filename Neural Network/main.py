import numpy as np
import pandas as pd
import sys
import plot
###############################################################################
path = 'data2Class_adjusted.txt'

###############################################################################


def read_data(path):
    data = pd.read_csv(path, sep=' ', header=None, names=['x0', 'x1', 'x2', 'class labels'])
    return data


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derv(z):
    out = sigmoid(z)
    return out * (1 - out)


neurons = 100


def rand_init_w():
    np.random.seed(1)
    w0 = 2 * np.random.random((neurons, 2)) - 1
    w0 = np.column_stack([np.ones(neurons), w0])  # for bias
    w0 = w0.T
    w1 = 2 * np.random.random((neurons, 1)) - 1
    return w0, w1


def forward(x, w):
    w1 = w[0]  # (100,3)
    w2 = w[1]  # (100, 1)

    L0_in = x  # 100 x 3 with bias at 1st col
    L0_out = x  # 100 x 3 with bias at 1st col

    L1_in = np.dot(L0_in, w1)  # (200,3) x (3,100) = (200, 100)
    L1_out = sigmoid(L1_in)  # (200, 100)

    L2_in = np.dot(L1_out, w2)  # (200,100) x (100,1) = (200, 1)
    #L2_out = sigmoid(L2_in) # (200, 1)
    L2_out = L2_in  # (200, 1)

    layer_inputs = [L0_in, L1_in, L2_in]
    layer_outputs = [L0_out, L1_out, L2_out]
    prediction = L2_out.copy()
    prediction[L2_out >= 0] = 1
    prediction[L2_out < 0] = -1

    return layer_inputs, layer_outputs, prediction


def backward_hinge(layer_inputs, layer_outputs, w, y):
    w1 = w[0]
    w2 = w[1]

    L0_in = layer_inputs[0]  # (200, 100)
    L1_in = layer_inputs[1]  # (200, 100)
    L2_in = layer_inputs[2]  # (200, 1)

    L0_out = layer_outputs[0]  # (200, 100)
    L1_out = layer_outputs[1]  # (200, 100)
    L2_out = layer_outputs[2]  # (200, 1)

    # cal output layer delta
    arg1 = np.zeros(y.shape[0]).reshape(y.shape[0], 1)
    arg2 = np.ones(y.shape[0]).reshape(y.shape[0], 1) - L2_out * y
    L2_err = np.maximum(arg1, arg2)

    L2_delta = np.zeros(y.shape[0]).reshape(y.shape[0], 1)
    for i, val in enumerate(arg2):
        if val > 0:
            L2_delta[i] = -1 * y[i]

    # cal hidden layer delta
    L1_err = np.dot(L2_delta, w2.T)
    L1_delta = L1_err * sigmoid_derv(L1_in)

    error = np.mean(L2_err ** 2)

    return error, L1_delta, L2_delta


def optimize_w(w, layer_outputs, L1_delta, L2_delta, alpha=0.05):
    w1 = w[0]  # 100,3
    w2 = w[1]  # 100,1

    L0_out = layer_outputs[0]  # (200, 1)
    L1_out = layer_outputs[1]  # (200, 100)
    L2_out = layer_outputs[2]  # (200, 1)

    w2 = w2 - alpha * np.dot(L1_out.T, L2_delta)
    w1 = w1 - alpha * np.dot(L0_out.T, L1_delta)

    w = [w1, w2]
    return w


###############################################################################
def main():
    data = read_data(path)
    x = data.iloc[:, 0:3]
    x = x.as_matrix()
    y = data.iloc[:, 3]
    y = y.as_matrix()
    y = y.reshape(200, 1)
    #plot.plot_input(x)
    plot.plot_inp_out(x,data.iloc[:, 3])
    #sys.exit()
    w0, w1 = rand_init_w()
    #w0 = np.zeros(w0.size).reshape(w0.shape)
    #w1 = np.zeros(w1.size).reshape(w1.shape)
    w = [w0, w1]

    iter = 1
    minErr = 0.02
    err = 10
    while err > minErr:
        layer_inputs, layer_outputs, pred = forward(x, w)
        #err, L1_delta, L2_delta = backward(layer_inputs, layer_outputs, w, y)
        err, L1_delta, L2_delta = backward_hinge(layer_inputs, layer_outputs, w, y)
        w = optimize_w(w, layer_outputs, L1_delta, L2_delta)
        #w = optimize_w(w, layer_outputs, L1_delta, L2_delta,0.05)
        if iter % 1000 == 0:
            print("iter:{0} \t error: {1}".format(iter, err))
        iter += 1

    plot.plot_predictions_2d(sigmoid(layer_outputs[2]))
    plot.plot_predictions_3d(x, sigmoid(layer_outputs[2]))
    test_data = [1, -1, -2]
    layer_inputs, layer_outputs, pred = forward(test_data, w)
    print('Prediction:', pred)


if __name__ == "__main__":
    main()
