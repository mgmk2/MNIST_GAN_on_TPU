import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from GANModel import GANModel

class GAN(object):
    def __init__(self, params, use_tpu=False):
        self.params = params
        self.use_tpu = use_tpu

        # データセットのロード
        self.X_train = self.load_dataset()
        self.num_batches = self.X_train.shape[0] // self.params.batch_size # ミニバッチの数
        print('number of batches:', self.num_batches)
        self.model = GANModel(params, use_tpu=use_tpu)

    def tpu_decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.use_tpu:
                with self.model.strategy.scope():
                    output = func(self, *args, **kwargs)
            else:
                output = func(self, *args, **kwargs)
            return output
        return wrapper

    def load_dataset(self):
        # mnistデータの読み込み
        (X_train, _), (_, _) = mnist.load_data()
        # 値を-1 to 1に規格化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        return X_train

    @tpu_decorator
    def fit(self):
        # TPU上でDiscriminatorとGeneratorを更新する

        start_fit = time.time()
        noise = np.random.normal(0, 1, (self.params.batch_size, self.params.z_dim)).astype(np.float32) # Generatorの入力
        image_real = self.X_train[:self.params.batch_size] # Discriminatorの入力

        # 入力パイプラインを初期化
        self.model.sess.run(
            self.model.iterator_init,
            feed_dict={
                self.model.images_placeholder: image_real,
                self.model.noise_placeholder: noise
            })

        # 学習前のGeneratorの出力を確認
        image_fake = self.model.sess.run(self.model.output_gen_ops)
        self.show_images(image_fake, epoch=0)

        # 学習開始
        for epoch in range(self.params.epochs):

            # 各エポックのコストと精度
            d_loss_epoch = 0
            d_acc_epoch = 0
            g_loss_epoch = 0
            g_acc_epoch = 0

            start_epoch = time.time()

            # 各エポックの学習前に学習データをシャッフル
            np.random.shuffle(self.X_train)

            # ミニバッチ学習
            for iter in range(self.num_batches):
                noise = np.random.normal(0, 1, (self.params.batch_size, self.params.z_dim)).astype(np.float32) # Generatorの入力
                image_real = self.X_train[iter * self.params.batch_size:(iter + 1) * self.params.batch_size] # Discriminatorの入力(本物)

                # iteratorを初期化
                self.model.sess.run(
                    self.model.iterator_init,
                    feed_dict={
                        self.model.images_placeholder: image_real, # Discriminatorの入力(本物)
                        self.model.noise_placeholder: noise # Genratorの入力
                    })

                #---------------------
                # Discriminatorの学習
                #---------------------

                # 本物画像でDiscriminatorを学習
                d_loss_real, d_acc_real = self.model.sess.run(self.model.train_disc_real_ops)
                # 偽物画像でDiscriminatorを学習
                d_loss_fake, d_acc_fake = self.model.sess.run(self.model.train_disc_fake_ops)

                # 本物画像の結果と偽物画像の結果を平均
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_acc = 0.5 * (d_acc_real + d_acc_fake)

                #---------------------
                # Generatorの学習
                #---------------------

                # 本物ラベルでGeneratorを学習
                g_loss, g_acc = self.model.sess.run(self.model.train_gen_ops)

                # エポック毎の結果
                d_loss_epoch += d_loss
                d_acc_epoch += d_acc
                g_loss_epoch += g_loss
                g_acc_epoch += g_acc

                # 進捗の表示
                sys.stdout.write(
                    '\repoch:{:d}  iter:{:d}   [D loss: {:f}, acc: {:.2f}%] [G loss: {:f}, acc: {:.2f}%]   '.format(
                        epoch + 1, iter + 1, d_loss, 100 * d_acc, g_loss, 100 * g_acc))
                sys.stdout.flush()

            # ミニバッチ毎の結果を平均
            d_loss_epoch /= self.num_batches
            d_acc_epoch /= self.num_batches
            g_loss_epoch /= self.num_batches
            g_acc_epoch /= self.num_batches

            epoch_time = time.time() - start_epoch

            # エポックの結果を表示
            sys.stdout.write(
                '\repoch:{:d}  iter:{:d}   [D loss: {:f}, acc: {:.2f}%] [G loss: {:f}, acc: {:.2f}%]   time: {:f}\n'.format(
                    epoch + 1, iter + 1, d_loss_epoch, 100 * d_acc_epoch, g_loss_epoch, 100 * g_acc_epoch, epoch_time))
            sys.stdout.flush()

            # Generatorの出力を確認
            if (epoch + 1) % 10 == 0:
                noise = np.random.normal(0, 1, (self.params.batch_size, self.params.z_dim)).astype(np.float32)
                self.model.sess.run(
                    self.model.iterator_init,
                    feed_dict={
                        self.model.images_placeholder: image_real, # Discriminatorの入力(使わないのでなんでもいい)
                        self.model.noise_placeholder: noise # Genratorの入力
                    })
                image_fake = self.model.sess.run(self.model.output_gen_ops)
                self.show_images(image_fake, epoch=epoch + 1)

    def show_images(self, images, epoch):
        # 出力画像を確認

        fig = plt.figure(figsize=(4, 4))
        for i in range(16):
          plt.subplot(4, 4, i + 1)
          plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
          plt.axis('off')

        fig.suptitle('epoch: {:}'.format(epoch))
        fig.savefig('mnist_epoch_{:}.png'.format(epoch))
        plt.show()
