seed = 2                        # random seed for numpy and tensorflow
pltshow = False                # If true, plot in Pycharm, if false, print visualizations to file
# Choose which files to output, to save memory, just plot contour
pltsurface = False
pltcontour = True
pltloss = False
plt_show_history = 1            # Choose what ratio of generator history data to plot (different from beta)
plt_sizeofhistorydots = 0.01    # Size of the dots for history data (set small to see better)

data = "line"        # Choose the type of data (see data folder)
gan_type = "gan"     # GAN type: normal GAN or WGAN
clipw = 3            # weight clipping value for critic for WGAN
Niter = 10000        # Total no. of iterations over discriminator-generator pair
g_lr = 0.01          # Generator learning rate
d_lr = 0.001         # Discriminator learning rate
d_l2_reg = 0.001     # (**NEW**) L2 weight regularization for discriminator
print_iter = 10      # Interval for printing visualization

N_G = 1             # No. of iterations to optimize generator
N_D = 100           # No. of iterations to optimize discriminator
ZN = 500            # No. of generated samples per batch
beta = 0.99            # AGAN decay rate (1 - keep all history of generated data, 0 - keep no history)
