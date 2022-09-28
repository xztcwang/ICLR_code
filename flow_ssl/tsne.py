import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

#digits=datasets.load_digits(n_class=6)
#X,y=digits.data, digits.target
#n_samples,n_features=X.shape

# n=20
# img=np.zeros((10*n,10*n))
# for i in range(n):
#     ix=10*i+1
#     for j in range(n):
#         iy=10*j+1
#         img[ix:ix+8,iy:iy+8]=X[i*n+j].reshape((8,8))
# plt.figure(figsize=(8,8))
# plt.imshow(img,cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# tsne=manifold.TSNE(n_components=2,init='pca',random_state=501)
# X_tsne=tsne.fit_transform(X)
# print("Original data dimension is {}. Embedded data dimension is {}".format(X.shape[-1],X_tsne.shape[-1]))
#
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne-x_min)/(x_max-x_min)
# plt.figure(figsize=(8,8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i,0],X_norm[i,1],str(y[i]),color=plt.cm.Set1(y[i]),
#              fontdict={'weight':'bold','size':9})
# plt.xticks([])
# plt.yticks([])
# plt.show()


def plot_tsne(Z, labels, centroids,step=None):
    #n_samples, n_features = Z.shape
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    Z_tsne = tsne.fit_transform(Z)
    z_min, z_max = Z_tsne.min(0), Z_tsne.max(0)
    Z_norm = (Z_tsne - z_min) / (z_max - z_min)

    centroids_tsne = tsne.fit_transform(centroids)
    centroids_min, centroids_max = centroids_tsne.min(0), centroids_tsne.max(0)
    centroids_norm = (centroids_tsne - centroids_min) / (centroids_max - centroids_min)

    for i in range(Z_norm.shape[0]):
        plt.scatter(Z_norm[i, 0], Z_norm[i, 1],
                    color=plt.cm.Set1(labels[i]),s=3)
    # for i in range(centroids_norm.shape[0]):
    #     plt.scatter(centroids_norm[i, 0], centroids_norm[i, 1],marker='D',
    #                 color='black', s=8)


    # for i in range(Z_norm.shape[0]):
    #     plt.text(Z_norm[i, 0], Z_norm[i, 1], str(labels[i]),
    #              color=plt.cm.Set1(labels[i]),
    #              s=3)


    # plt.figure(figsize=(10, 10))
    # for i in range(Z_norm.shape[0]):
    #     plt.text(Z_norm[i, 0], Z_norm[i, 1], str(labels[i]),
    #              color=plt.cm.Set1(labels[i]),
    #              fontdict={'weight': 'bold', 'size': 9})
    # for i in range(centroids_norm.shape[0]):
    #     plt.text(Z_norm[i, 0], Z_norm[i, 1], str(i),
    #              color='black',
    #              fontdict={'weight': 'bold', 'size': 14})
    plt.xticks([])
    plt.yticks([])
    #titlestr = 't-SNE at epoch ' + str(step)
    # if whichdata=='raw':
    #     plt.title('t-SNE of Raw Data (Cora)')
    # else:
    #     plt.title(titlestr)
    plt.savefig("/home/tkw5356/GCFlow/experiments/train_flows/save/tSNE_flowgmm_wikics.png")
    #plt.show()
    # plt.figure(figsize=(10, 10))
    # for i in range(Z_norm.shape[0]):
    #     plt.text(Z_norm[i, 0], Z_norm[i, 1], str(labels[i]),
    #              color=plt.cm.Set1(labels[i]),
    #              fontdict={'weight': 'bold', 'size': 9})
    # plt.savefig("/home/tkw5356/GCFlow/experiments/train_flows/save/tSNE_flowgmm_cora2.png")
