import numpy as np

seed = 2                            # random seed for numpy and tensorflow
z_dim = 500
batchnorm = True
train_size = 64
# set the gan type here, options are "gan" and "wgan"
gan_type = "gan"                    # GAN type: normal GAN or WGAN (WGAN not implemented yet)
Nepoch = 100                        # Total no. of iterations over discriminator-generator pair
batch_size = 32
sample_size = batch_size            # Sample size to estimate gloss, dloss for printout
Ncropsamples = 25                    # No. of generated samples to display
g_lr = 0.0002                       # Generator learning rate
d_lr = 0.0002                       # Discriminator learning rate
N_G = 1                             # No. of iterations to optimize generator
N_D = 100                             # No. of iterations to optimize discriminator
beta = 0                         # AGAN decay rate (1 - keep all history of generated data, 0 - keep no history)

print_iter = 10                     # Interval for printing visualization
adam_param = 0.5                    # Adam optimizer beta1 parameter (don't confuse with our AGAN beta)

# Data details
data = 'lsun'
#data_dir='/home/koukl/mnist'
#sample_dir='/home/koukl/dcgan1/samples'
if data == 'mnist':
    data_dir='/Users/connie/PycharmProjects/accumulative-GAN_AGAN/data'
    image_len = image_height = image_width = 28
    image_dims = [image_height, image_width, 1]
    extract_digit = 4  # If data = 'mnist', choose one of digit 0-9 to learn
elif data == 'lsun':
    data_dir='/Users/connie/PycharmProjects/lsun-master/lsun_train'
    image_len = image_height = image_width = 64
    image_dims = [image_height, image_width, 3]

sample_dir='/Users/connie/PycharmProjects/accumulative-GAN_AGAN/20171130_DCAGAN/samples'

# conv network details
d_fm = 64                           # Discriminator number of feature maps in conv layers
g_fm = 64                           # Generator number of feature maps in conv layers

