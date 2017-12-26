import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.sparse.linalg as sla
import numpy as np
import os

########################################################################################################################
directory = 'D:\\iNFOTECH\\MachineLearning\\Exercises\\ex05\\yalefaces'


# load the data
# x = plt.imread('yalefaces\yalefaces\yalefaces\subject01.gif')
# plt.imshow(x)
# plt.show()

def read_images(x):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # print(os.path.join(directory, filename))
            continue
        else:
            temp = np.array(plt.imread(directory + '\\' + filename))
            temp = np.reshape(temp, (1, 77760))
            x[count] = temp
            count += 1
    shape = np.array(plt.imread(directory + '\\' + filename)).shape
    return x, shape


def cal_mean_face(x):
    mean_face = np.mean(x, axis=0)
    return np.matrix(mean_face).T


def matrix_center(x, mean_face):
    ones = np.ones((166, 1))
    x = x - ones.dot(mean_face.T)
    return x


def cal_svd(x, num_eigenvalues):
    u, s, vt = sla.svds(x, k=num_eigenvalues)
    return vt.T


def reduce_dimensions(x, v):
    return x.dot(v)


def reconstruct_image(mean_face, z, v):
    ones = np.ones((166, 1))
    return ones.dot(mean_face.T) + z.dot(v.T)

def cal_reconstruct_error(x,x_recon):
    t = x - x_recon
    t = np.power(t, 2)
    return t.sum()

def plot_x_recon (x, shape):
    #for i in range(x.shape[0]):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    y = plt.imread(directory + '\\' + 'subject15.wink')
    plt.imshow(y)
    a.set_title('Before')
    a = fig.add_subplot(1, 2, 2)
    img = x[165, :]
    img = np.reshape(img, shape)
    plt.imshow(img)
    a.set_title('After')
    plt.show()

x = np.empty((166, 77760))
x, image_shape = read_images(x)
print("image_shape:", image_shape)
print("Input X: ", x)
print("x_shape:", x.shape)

mean_face = cal_mean_face(x)
x_cen = matrix_center(x, mean_face)
print("x_cen: ", x_cen)
print("x_cen_shape:", x_cen.shape)

p = 70  # output dimension
v = cal_svd(x_cen, p)
print("v_shape:", v.shape)
z = reduce_dimensions(x_cen, v)
print("z_shape:", z.shape)

x_recon = reconstruct_image(mean_face, z, v)
print("x_recon  _shape:", x_recon.shape)

error = cal_reconstruct_error(x, x_recon)
print("error:", error)

plot_x_recon(x_recon, image_shape)