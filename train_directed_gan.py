#!/usr/bin/python
# -*- coding: utf-8 -*-


import argparse
import os

import chainer
import chainer.functions as F
import chainer.links as L


from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList

from chainer import training
from chainer.training import extensions

from net_mnist import Discriminator
from net_mnist import Generator
from updater import DirectedGANUpdater
from visualize import out_generated_image


def main():
    parser = argparse.ArgumentParser(description='Chainer: DCGAN MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--optimizer', '-p', type=str, default='Adam',
                        help='optimizer')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # まず同じネットワークのオブジェクトを作る.
    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()


    #model = L.Classifier(MLP(args.unit, 10))



    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    '''
    こんなふうにargs.optimizerによって、選択出来る
    
    if args.optimizer == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad()
    elif args.optimizer == 'SGD':
        optimizer = chainer.optimizers.SGD()
    elif args.optimizer == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD()
    elif args.optimizer == 'RMSprop':
        optimizer = chainer.optimizers.RMSprop()
    else:
        optimizer = chainer.optimizers.Adam()
    '''



    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Load the MNIST dataset
    train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=3, scale=255.) # ndim=3 : (ch,width,height)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    #test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)



    # Set up a trainer
    updater = DirectedGANUpdater(models=(gen, dis), iterator=train_iter, optimizer={'gen':opt_gen, 'dis':opt_dis}, device=args.gpu)

    #updater = training.updaters.StandardUpdater(train_iter, optimizer, device = args.gpu, loss_func = model.get_loss_func())

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)



    '''
    デフォルトでは、1epochごとにlogにlossやaccuracyが記録されるが、
    1epoch時点で例えばかなりlossが減少しているな、という時、
    こんなふうにすると：
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    これで100iterationごとに記録される。
    
    '''





    epoch_interval = (1, 'epoch')
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    # trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.snapshot_object(dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)

    #trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    # test_iterを使ってepoch毎に評価している

    # trainer.extend(extensions.dump_graph('main/loss'))
    # ネットワークの形をグラフで表示できるようにdot形式で保存する. main/lossは答えとの差の大きさ.

    # trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    # epochごとのtrainerの情報を保存する.それを読み込んで、途中から再開などができる.


    trainer.extend(extensions.LogReport(trigger=display_interval))
    # epochごとにlogをだす.


    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'gen/loss', 'dis/loss', 'reconstruction/loss']), trigger=display_interval)
    # logで出す情報を指定する. reconstruction/lossを追加して書いたが、gen/lossの代わりに書いてもいいかも.

    #trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    # main/lossは答えとの差の大きさ.  main/accuracyは正解率.

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_generated_image(gen, dis, 10, 10, args.seed, args.out), trigger=epoch_interval)
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()

