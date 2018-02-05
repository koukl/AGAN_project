import tensorflow as tf
import numpy      as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import config
import scipy.misc

'''
def visualize(gout, dgridout, xpos, gridxy, z, zfeed, xfeed, sess, i, alldloss, allgloss):
    pltshow = True
    matplotlib.rcParams.update({'font.size': 15})

    # plot out generated samples vs real samples
    a = sess.run(gout, feed_dict={xpos: xfeed, z: zfeed})

    # plot discriminator's score over grid
    gridbins = 100
    x = np.linspace(0, 1, gridbins)
    y = np.linspace(0, 1, gridbins)
    xv, yv = np.meshgrid(x, y)
    gridsample = np.array([xv.ravel(), yv.ravel()]).T
    gridscore = sess.run(dgridout, feed_dict={xpos: xfeed, gridxy: gridsample})

    fig = plt.figure(1, figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xfeed[:, 0], xfeed[:, 1], c='r', marker='o', s=0.2, label='real data')
    ax.scatter(a[:, 0], a[:, 1], c='b', marker='^', s=0.2, label='generated data')
    ax.scatter(gridsample[:, 0], gridsample[:, 1], gridscore, c='g', marker='.', s=0.2, label='discriminator score')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('D score')
    ax.set_zlim3d(0, 1)
    #ax.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.grid(True)

    if pltshow:
        plt.show()
    else:
        plt.savefig('success/surface' + str(i).zfill(4) + '.png')
    plt.close()

    #######################################################
    if pltshow == False:
        plt.figure(1, figsize=(6, 6))
        cax = plt.contourf(xv, yv, gridscore.reshape(xv.shape), cmap='coolwarm', alpha=0.7)
        plt.scatter(xfeed[:, 0], xfeed[:, 1], c='r', marker='o', s=5, label='real data')
        plt.scatter(a[:, 0], a[:, 1], c='b', marker='^', s=5, label='generated data')
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        #fig.colorbar(cax)
        if pltshow:
            plt.show()
        else:
            plt.savefig('success/contour' +  str(i).zfill(4) + '.png')
        plt.close()

    #######################################################
    if pltshow == False:
        plt.figure(1, figsize=(10, 6))
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0))

        xd = np.arange(alldloss.shape[0])
        xg = np.arange(allgloss.shape[0])
        ax1.plot(xd, alldloss)
        ax1.set_ylabel('dloss')

        #ax = fig.add_subplot(313)
        ax2.plot(xg, allgloss)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('gloss')

        if pltshow:
            plt.show()
        else:
            plt.savefig('success/loss' +  str(i).zfill(4) + '.png')
        #plt.grid(True)



    plt.close()
'''
'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xfeed[:, 0], xfeed[:, 1], c='r', marker='o', s=0.2, label='real data')
    ax.scatter(a[:, 0], a[:, 1], c='b', marker='^', s=0.2, label='generated data')
    ax.scatter(gridsample[:, 0], gridsample[:, 1], gridscore, c='g', marker='.', s=0.2, label='discriminator score')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Dscore')
    ax.set_zlim3d(0, 1)
    ax.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.grid(True)

    plt.show()
    plt.close()
'''

def save_loss(alldloss, allgloss, filename):

    plt.figure(1, figsize=(10, 6))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))

    xd = np.arange(alldloss.shape[0])
    xg = np.arange(allgloss.shape[0])
    ax1.plot(xd, alldloss)
    ax1.set_ylabel('dloss')

    # ax = fig.add_subplot(313)
    ax2.plot(xg, allgloss)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('gloss')
    plt.savefig(filename)
    plt.close()
        # plt.grid(True)


def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)


def inverse_transform(images):
  return (images+1.)/2.


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w




