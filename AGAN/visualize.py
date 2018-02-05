import tensorflow as tf
import numpy      as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
import config

'''
def make_views(ax, angles, elevation=None, width=4, height=3,
               prefix='tmprot_', **kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width, height)

    for i, angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = '%s%03d.jpeg' % (prefix, i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files

def make_gif(files, output, delay=100, repeat=True, **kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              % (delay, loop, " ".join(files), output))

def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """
    output_ext = os.path.splitext(output)[1]

    files = make_views(ax, angles, **kwargs)

    D = {'.gif': make_gif}

    D[output_ext](files, output, **kwargs)

    for f in files:
        os.remove(f)
'''

def visualize(gout, h1, h2, dgridout, xpos, gridxy, z, zfeed, xfeed, sess, i, alldloss, allgloss, pltshow, accum_gout_):
    #matplotlib.rcParams.update({'font.size': 15})

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
    ax.scatter(accum_gout_[:, 0], accum_gout_[:, 1], c='m', marker='.', s=0.2, label='older generated data')
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
        if config.pltsurface:
            plt.savefig('success/surface' + str(i).zfill(4) + '.png')
    plt.close()

    #######################################################
    if pltshow == False:
        plt.figure(1, figsize=(6, 6))
        cax = plt.contourf(xv, yv, gridscore.reshape(xv.shape), cmap='coolwarm', alpha=0.7)
        np.random.shuffle(accum_gout_)
        Nretain = int(config.plt_show_history * accum_gout_.shape[0])
        hist_gen_to_plot = accum_gout_[:Nretain]
        plt.scatter(hist_gen_to_plot [:, 0], hist_gen_to_plot [:, 1], c='m', marker='.', s=config.plt_sizeofhistorydots, label='older generated data')
        plt.scatter(xfeed[:, 0], xfeed[:, 1], c='r', marker='o', s=5, label='real data')
        plt.scatter(a[:, 0], a[:, 1], c='b', marker='^', s=5, label='generated data')
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        #plt.colorbar(cax)
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                        mode="expand", borderaxespad=0, ncol=3)
        if pltshow:
            plt.show()
        else:
            if config.pltcontour:
                plt.savefig('plots/contour' +  str(i).zfill(4) + '.png')
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
            if config.pltloss:
                plt.savefig('plots/loss' +  str(i).zfill(4) + '.png')
        #plt.grid(True)



    plt.close()


'''
def visualize_manifold_hidden2(gout, h1, h2, dgridout, xpos, gridxy, z, zfeed, xfeed, sess, iter, alldloss, allgloss, pltshow):
    matplotlib.rcParams.update({'font.size': 15})
    # plot out generated samples vs real samples
    gout_, h1_, h2_ = sess.run([gout, h1, h2], feed_dict={xpos: xfeed, z: zfeed})

    # generated over grid
    gridbins = 10
    x = np.linspace(0, 1, gridbins)
    y = np.linspace(0, 1, gridbins)
    xv, yv = np.meshgrid(x, y)
    gridsample = np.array([xv.ravel(), yv.ravel()]).T
    #gridscore = sess.run(dgridout, feed_dict={xpos: xfeed, gridxy: gridsample})
    goutgrid_, h1grid_, h2grid_ = sess.run([gout, h1, h2], feed_dict={xpos: xfeed, z: gridsample})

    plt.figure(1, figsize=(24, 6))
    plt.subplot(141)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    #plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(zfeed[:, 0], zfeed[:, 1])
    ngridlines = xv.shape[0]
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(xv[:, i], yv[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(xv[i, :], yv[i, :], c='0.75')
    #plt.colorbar()

    plt.subplot(142)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('h1')
    plt.ylabel('h2')
    # plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(h1_[:, 0], h1_[:, 1])
    ngridlines = xv.shape[0]
    h1grid0 = h1grid_[:, 0].reshape(xv.shape)
    h1grid1 = h1grid_[:, 1].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(h1grid0[:, i], h1grid1[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(h1grid0[i, :], h1grid1[i, :], c='0.75')


    plt.subplot(143)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('h3')
    plt.ylabel('h4')
    # plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(h2_[:, 0], h2_[:, 1])
    ngridlines = xv.shape[0]
    h2grid0 = h2grid_[:, 0].reshape(xv.shape)
    h2grid1 = h2grid_[:, 1].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(h2grid0[:, i], h2grid1[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(h2grid0[i, :], h2grid1[i, :], c='0.75')

    plt.subplot(144)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('o1')
    plt.ylabel('o2')
    # plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(gout_[:, 0], gout_[:, 1])
    plt.scatter(xfeed[:, 0], xfeed[:, 1])
    ngridlines = xv.shape[0]
    goutgrid0 = goutgrid_[:, 0].reshape(xv.shape)
    goutgrid1 = goutgrid_[:, 1].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(goutgrid0[:, i], goutgrid1[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(goutgrid0[i, :], goutgrid1[i, :], c='0.75')

    if pltshow:
        plt.show()
    else:
        plt.savefig('success_line/' + str(iter).zfill(4) + '.png')
    plt.close()



def visualize_manifold_hidden3_x2(gout, h1, h2, dgridout, xpos, gridxy, z, zfeed, xfeed, sess, iter, alldloss, allgloss, pltshow, elev=10, azim=120):
    matplotlib.rcParams.update({'font.size': 13})
    # plot out generated samples vs real samples
    Ndata_toshow = 50
    zfeed = zfeed[:Ndata_toshow,:]
    gout_, h1_, h2_ = sess.run([gout, h1, h2], feed_dict={xpos: xfeed, z: zfeed})

    # generated over grid
    gridbins = 10
    x = np.linspace(0, 1, gridbins)
    y = np.linspace(0, 1, gridbins)
    xv, yv = np.meshgrid(x, y)
    gridsample = np.array([xv.ravel(), yv.ravel()]).T
    #gridscore = sess.run(dgridout, feed_dict={xpos: xfeed, gridxy: gridsample})
    goutgrid_, h1grid_, h2grid_ = sess.run([gout, h1, h2], feed_dict={xpos: xfeed, z: gridsample})

    fig = plt.figure(1, figsize=(25, 5))
    plt.subplot(141)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    #plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(zfeed[:, 0], zfeed[:, 1])
    ngridlines = xv.shape[0]
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(xv[:, i], yv[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(xv[i, :], yv[i, :], c='0.75')
    #plt.colorbar()

    ax = fig.add_subplot(142, projection='3d')
    # hidden node activation is sigmoid, hence [0,1]
    ax.set_xlabel('h1')
    ax.set_ylabel('h2')
    ax.set_zlabel('h3')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    # Customize the view angle so it's easier to see
    ax.view_init(elev=elev, azim=azim)
    ngridlines = xv.shape[0]
    h1grid_0 = h1grid_[:, 0].reshape(xv.shape)
    h1grid_1 = h1grid_[:, 1].reshape(xv.shape)
    h1grid_2 = h1grid_[:, 2].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        ax.plot(h1grid_0[:, i], h1grid_1[:, i], h1grid_2[:, i], c='0.75', lw=1.5)
        # horizontal, yy fixed
        ax.plot(h1grid_0[i, :], h1grid_1[i, :], h1grid_2[i, :], c='0.75', lw=1.5)
    ax.scatter(h1_[:, 0], h1_[:, 1], h1_[:, 2])

    ax = fig.add_subplot(143, projection='3d')
    # hidden node activation is sigmoid, hence [0,1]
    ax.set_xlabel('h4')
    ax.set_ylabel('h5')
    ax.set_zlabel('h6')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    # Customize the view angle so it's easier to see
    ax.view_init(elev=elev, azim=azim)
    ngridlines = xv.shape[0]
    h2grid_0 = h2grid_[:, 0].reshape(xv.shape)
    h2grid_1 = h2grid_[:, 1].reshape(xv.shape)
    h2grid_2 = h2grid_[:, 2].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        ax.plot(h2grid_0[:, i], h2grid_1[:, i], h2grid_2[:, i], c='0.75', lw=1.5)
        # horizontal, yy fixed
        ax.plot(h2grid_0[i, :], h2grid_1[i, :], h2grid_2[i, :], c='0.75', lw=1.5)
    ax.scatter(h2_[:, 0], h2_[:, 1], h2_[:, 2])

    plt.subplot(144)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('o1')
    plt.ylabel('o2')
    # plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(gout_[:, 0], gout_[:, 1])
    plt.scatter(xfeed[:, 0], xfeed[:, 1])
    ngridlines = xv.shape[0]
    goutgrid0 = goutgrid_[:, 0].reshape(xv.shape)
    goutgrid1 = goutgrid_[:, 1].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(goutgrid0[:, i], goutgrid1[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(goutgrid0[i, :], goutgrid1[i, :], c='0.75')

    if pltshow:
        plt.show()
    else:
        plt.savefig('success_line_manifold/' + str(iter).zfill(4) + '.png')
    plt.close()



def visualize_manifold_hidden3_x1(gout, h1, h2, dgridout, xpos, gridxy, z, zfeed, xfeed, sess, iter, alldloss, allgloss, pltshow, elev=10, azim=120):
    matplotlib.rcParams.update({'font.size': 13})
    # plot out generated samples vs real samples
    Ndata_toshow = 50
    zfeed = zfeed[:Ndata_toshow,:]
    gout_, h1_, h2_ = sess.run([gout, h1, h2], feed_dict={xpos: xfeed, z: zfeed})

    # generated over grid
    gridbins = 10
    x = np.linspace(0, 1, gridbins)
    y = np.linspace(0, 1, gridbins)
    xv, yv = np.meshgrid(x, y)
    gridsample = np.array([xv.ravel(), yv.ravel()]).T
    #gridscore = sess.run(dgridout, feed_dict={xpos: xfeed, gridxy: gridsample})
    goutgrid_, h1grid_, h2grid_ = sess.run([gout, h1, h2], feed_dict={xpos: xfeed, z: gridsample})

    fig = plt.figure(1, figsize=(25, 5))
    plt.subplot(131)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    #plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(zfeed[:, 0], zfeed[:, 1])
    ngridlines = xv.shape[0]
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(xv[:, i], yv[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(xv[i, :], yv[i, :], c='0.75')
    #plt.colorbar()

    ax = fig.add_subplot(132, projection='3d')
    # hidden node activation is sigmoid, hence [0,1]
    ax.set_xlabel('h1')
    ax.set_ylabel('h2')
    ax.set_zlabel('h3')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    # Customize the view angle so it's easier to see
    ax.view_init(elev=elev, azim=azim)
    ngridlines = xv.shape[0]
    h1grid_0 = h1grid_[:, 0].reshape(xv.shape)
    h1grid_1 = h1grid_[:, 1].reshape(xv.shape)
    h1grid_2 = h1grid_[:, 2].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        ax.plot(h1grid_0[:, i], h1grid_1[:, i], h1grid_2[:, i], c='0.75', lw=1.5)
        # horizontal, yy fixed
        ax.plot(h1grid_0[i, :], h1grid_1[i, :], h1grid_2[i, :], c='0.75', lw=1.5)
    ax.scatter(h1_[:, 0], h1_[:, 1], h1_[:, 2])

    plt.subplot(133)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('o1')
    plt.ylabel('o2')
    # plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(gout_[:, 0], gout_[:, 1])
    plt.scatter(xfeed[:, 0], xfeed[:, 1])
    ngridlines = xv.shape[0]
    goutgrid0 = goutgrid_[:, 0].reshape(xv.shape)
    goutgrid1 = goutgrid_[:, 1].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(goutgrid0[:, i], goutgrid1[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(goutgrid0[i, :], goutgrid1[i, :], c='0.75')

    if pltshow:
        plt.show()
    else:
        plt.savefig('success_line_manifold/' + str(iter).zfill(4) + '.png')
    plt.close()



def visualize_manifold_hidden3_x1_indiv(gout, h1, h2, dgridout, xpos, gridxy, z, zfeed, xfeed, sess, iter, alldloss, allgloss, pltshow, elev=10, azim=120):
    animate = True
    matplotlib.rcParams.update({'font.size': 13})
    # plot out generated samples vs real samples
    Ndata_toshow = 50
    zfeed = zfeed[:Ndata_toshow,:]
    gout_, h1_, h2_ = sess.run([gout, h1, h2], feed_dict={xpos: xfeed, z: zfeed})

    # generated over grid
    gridbins = 10
    x = np.linspace(0, 1, gridbins)
    y = np.linspace(0, 1, gridbins)
    xv, yv = np.meshgrid(x, y)
    gridsample = np.array([xv.ravel(), yv.ravel()]).T
    #gridscore = sess.run(dgridout, feed_dict={xpos: xfeed, gridxy: gridsample})
    goutgrid_, h1grid_, h2grid_ = sess.run([gout, h1, h2], feed_dict={xpos: xfeed, z: gridsample})

    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    #plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(zfeed[:, 0], zfeed[:, 1])
    ngridlines = xv.shape[0]
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(xv[:, i], yv[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(xv[i, :], yv[i, :], c='0.75')
    plt.savefig('success_line_manifold/'+ str(iter).zfill(4) + '_x.png')

    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('h1')
    ax.set_ylabel('h2')
    ax.set_zlabel('h3')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    # Customize the view angle so it's easier to see
    ax.view_init(elev=elev, azim=azim)
    ngridlines = xv.shape[0]
    h1grid_0 = h1grid_[:, 0].reshape(xv.shape)
    h1grid_1 = h1grid_[:, 1].reshape(xv.shape)
    h1grid_2 = h1grid_[:, 2].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        ax.plot(h1grid_0[:, i], h1grid_1[:, i], h1grid_2[:, i], c='0.75', lw=1.5)
        # horizontal, yy fixed
        ax.plot(h1grid_0[i, :], h1grid_1[i, :], h1grid_2[i, :], c='0.75', lw=1.5)
    ax.scatter(h1_[:, 0], h1_[:, 1], h1_[:, 2])
    if animate:
        angles = np.linspace(0, 360, 21)[:-1]  # Take 20 angles between 0 and 360
        # create an animated gif (20ms between frames)
        rotanimate(ax, angles, 'success_line_manifold/'+ str(iter).zfill(4) + '_h.gif', delay=50, elevation=elev, width=5, height=5)
    plt.clf()

    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('o1')
    plt.ylabel('o2')
    # plt.contourf(xx2, yy2, yop, cmap=cmap, alpha=0.7)
    plt.scatter(gout_[:, 0], gout_[:, 1])
    plt.scatter(xfeed[:, 0], xfeed[:, 1])
    ngridlines = xv.shape[0]
    goutgrid0 = goutgrid_[:, 0].reshape(xv.shape)
    goutgrid1 = goutgrid_[:, 1].reshape(xv.shape)
    for i in range(ngridlines):
        # plot vertical lines, keep xx fixed
        plt.plot(goutgrid0[:, i], goutgrid1[:, i], c='0.75')
        # horizontal, yy fixed
        plt.plot(goutgrid0[i, :], goutgrid1[i, :], c='0.75')
    plt.savefig('success_line_manifold/' + str(iter).zfill(4) + '_op.png')

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

