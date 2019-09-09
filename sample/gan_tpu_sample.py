import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential

class GAN(object):
    def __init__(self):

        self.z_dim = 100 # 潜在変数の次元

        self.image_shape = (28, 28, 1) # 画像のサイズ
        self.noise_shape = (self.z_dim,) # ノイズのサイズ

        self.epochs = 1000 # 学習回数
        self.batch_size = 512 # バッチサイズ

        # データセットのロード
        self.X_train = self.load_dataset()
        self.num_batches = self.X_train.shape[0] // self.batch_size # ミニバッチの数
        print('number of batches:', self.num_batches)

        # TPU対応のおまじない1
        tf.keras.backend.clear_session()
        tpu_grpc_url = "grpc://" + os.environ["COLAB_TPU_ADDR"]
        self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        self.strategy = tf.contrib.distribute.TPUStrategy(self.tpu_cluster_resolver)

        # ここからTPU対応のモデルやらを書いていく
        with self.strategy.scope():

            # Discriminatorの定義
            self.discriminator = self.build_discriminator()
            self.optimizer_disc = tf.train.AdamOptimizer(2.0e-4, 0.5) # Discriminator用のOptimizer
            self.var_disc = self.discriminator.trainable_variables # Discriminatorの重み

            # Generatorの定義
            self.generator = self.build_generator()
            self.optimizer_gen = tf.train.AdamOptimizer(2.0e-4, 0.5) # Generator用のOptimizer
            self.var_gen = self.generator.trainable_variables # Generatorの重み

            # データセットの入力用placeholder
            self.images_placeholder = tf.placeholder(tf.float32, [None, *self.image_shape])
            self.noise_placeholder = tf.placeholder(tf.float32, [None, *self.noise_shape])
            self.labels_placeholder = tf.placeholder(tf.float32, [None, 1])

            # Dataset APIで入力パイプラインを定義
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.images_placeholder,
                 self.noise_placeholder,
                 self.labels_placeholder
                ))
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size, drop_remainder=True) # TPUではdrop_remainder=Trueが必須

            # DatasetをTPU用のDatasetに変換
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)

            # iteratorを定義
            input_iterator = dist_dataset.make_initializable_iterator()
            self.iterator_init = input_iterator.initialize()

            # 学習等のopsを定義
            inputs = input_iterator.get_next() # ネットワークの入力
            self.train_disc_ops = self.train_step_disc(inputs) # Discriminatorの学習
            self.train_gen_ops = self.train_step_gen(inputs) # Generatorの学習
            self.output_gen_ops = self.output_images_gen(inputs) # Generatorの出力

            # TPU対応のおまじない2
            tf.contrib.distribute.initialize_tpu_system(self.tpu_cluster_resolver)
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            cluster_spec = self.tpu_cluster_resolver.cluster_spec()
            if cluster_spec:
                config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

            # Sessionの定義
            self.sess = tf.Session(
                target=self.tpu_cluster_resolver.master(),
                config=config
            )

            # 変数の初期化
            self.sess.run(tf.global_variables_initializer())

    def load_dataset(self):

        # mnistデータの読み込み
        (X_train, _), (_, _) = mnist.load_data()

        # 値を-1 to 1に規格化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        return X_train

    def build_discriminator(self):
        # discriminatorモデル
        # kerasのSequentialを使っているが、Functional APIでもtensorflowの低レベルAPIでもたぶん大丈夫

        layers_disc = []
        layers_disc.append(
            Conv2D(16, (5, 5), strides=(2, 2), padding='same', input_shape=self.image_shape))
        layers_disc.append(LeakyReLU(alpha=0.2))

        layers_disc.append(
            Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        layers_disc.append(LeakyReLU(alpha=0.2))

        layers_disc.append(Flatten())
        layers_disc.append(Dense(1))

        discriminator = Sequential(layers_disc)

        return discriminator

    def build_generator(self):
        # Generatorモデル
        # kerasのSequentialを使っているが、Functional APIでもtensorflowの低レベルAPIでもたぶん大丈夫

        layers_gen = []
        layers_gen.append(Dense(7 * 7 * 256, use_bias=False, input_shape=self.noise_shape))
        layers_gen.append(BatchNormalization(momentum=0.8))
        layers_gen.append(LeakyReLU(alpha=0.2))

        layers_gen.append(Reshape((7, 7, 256)))

        layers_gen.append(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        layers_gen.append(BatchNormalization(momentum=0.8))
        layers_gen.append(LeakyReLU(alpha=0.2))

        layers_gen.append(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        layers_gen.append(BatchNormalization(momentum=0.8))
        layers_gen.append(LeakyReLU(alpha=0.2))

        layers_gen.append(
            Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))

        generator = Sequential(layers_gen)

        return generator

    def train_step_disc(self, dist_inputs):
        # Discriminatorに対して
        # コストを計算して逆伝播法で重みを更新する

        def step_fn(inputs):
            features, _, labels = inputs # 入力データ
            logits = self.discriminator(features) # Discriminatorの出力

            # コスト関数と重み更新
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) / self.batch_size # reduce_meanは使わない方がいい
            train_op_disc = self.optimizer_disc.minimize(loss, var_list=self.var_disc) # discriminatorの重みのみ更新する

            # 精度
            logits_bool = tf.cast(tf.greater_equal(logits, 0), tf.float32)
            acc = tf.reduce_sum(1.0 - tf.abs(labels - logits_bool)) / self.batch_size

            # 必ずtf.control_dependenciesを使うこと
            with tf.control_dependencies([train_op_disc]):
                return tf.identity(loss), tf.identity(acc)

        # TPUコア毎にstep_fnを実行して結果を出力
        per_replica_losses, per_replica_accs = self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))

        # TPUコア毎のコストと精度をまとめる
        # tf.distribute.ReduceOp.SUMはtf.reduce_sum
        # tf.distribute.ReduceOp.MEANはtf.reduce_meanに対応
        # MEANは正しい結果になっているかちょっと自信ないので、SUMにしている
        losses = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        accs = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_accs, axis=None)

        return losses, accs

    def output_images_gen(self, dist_inputs):
        # Generatorの出力画像を得る

        def step_fn(inputs):
            _, noises, _ = inputs # 入力データ
            return self.generator(noises, training=False) # GeneratorにBatchNormalizationを入れている場合はtraining=Falseを指定

        # TPUコア毎にstep_fnを実行して結果を出力
        gen_output = self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))

        # TPUコア毎の結果を連結
        gen_output = tf.concat(gen_output.values, axis=0)

        return gen_output

    def train_step_gen(self, dist_inputs):
        # Generatorに対して
        # コストを計算して逆伝播法で重みを更新する

        def step_fn(inputs):
            _, noises, labels = inputs # 入力データ
            features = self.generator(noises, training=True) # GeneratorにBatchNormalizationを入れている場合はtraining=Trueを指定
            logits = self.discriminator(features)

            # コスト関数と重み更新
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) / self.batch_size
            train_op_gen = self.optimizer_gen.minimize(loss, var_list=self.var_gen) # Generatorの重みのみ更新

            # 精度
            logits_bool = tf.cast(tf.greater_equal(logits, 0), tf.float32)
            acc = tf.reduce_sum(1.0 - tf.abs(labels - logits_bool)) / self.batch_size

            # BatchNormalizationの平均と分散の更新
            # GeneratorにBatchNormalizationを入れている場合は必須
            update_ops = self.generator.get_updates_for(None) + self.generator.get_updates_for(noises)

            # 必ずtf.control_dependenciesを使うこと
            # BatchNormalizationを使っている場合はupdate_opsも一緒に入れる
            with tf.control_dependencies([train_op_gen, *update_ops]):
                return tf.identity(loss), tf.identity(acc)

        # TPUコア毎にstep_fnを実行して結果を出力
        per_replica_losses, per_replica_accs = self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))

        # TPUコア毎のコストと精度をまとめる
        # tf.distribute.ReduceOp.SUMはtf.reduce_sum
        # tf.distribute.ReduceOp.MEANはtf.reduce_meanに対応
        # MEANは正しい結果になっているかちょっと自信ないので、SUMにしている
        losses = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        accs = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_accs, axis=None)

        return losses, accs

    def fit(self):
        # TPU上でDiscriminatorとGeneratorを更新する

        with self.strategy.scope():
            start_fit = time.time()

            noise = np.random.normal(0, 1, (self.batch_size, self.z_dim)).astype(np.float32) # Generatorの入力
            image_real = self.X_train[:self.batch_size] # Discriminatorの入力
            label_real = np.ones((self.batch_size, 1), np.float32) # Discriminatorの出力ラベル

            # 入力パイプラインを初期化
            self.sess.run(
                self.iterator_init,
                feed_dict={
                    self.images_placeholder: image_real,
                    self.noise_placeholder: noise,
                    self.labels_placeholder: label_real
                })

            # 学習前のGeneratorの出力を確認
            image_fake = self.sess.run(self.output_gen_ops)
            self.show_images(image_fake, epoch=0)

            # 学習開始
            for epoch in range(self.epochs):

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

                    noise = np.random.normal(0, 1, (self.batch_size, self.z_dim)).astype(np.float32) # Generatorの入力
                    image_real = self.X_train[iter * self.batch_size:(iter + 1) * self.batch_size] # Discriminatorの入力(本物)
                    label_real = np.ones((self.batch_size, 1), np.float32) # Discriminatorの出力ラベル(本物)
                    label_fake = np.zeros((self.batch_size, 1), np.float32) # Discriminatorの出力ラベル(偽物)

                    #---------------------
                    # Discriminatorの学習
                    #---------------------

                    # iteratorを初期化
                    self.sess.run(
                        self.iterator_init,
                        feed_dict={
                            self.images_placeholder: image_real, # Discriminatorの入力(本物)
                            self.noise_placeholder: noise, # Genratorの入力
                            self.labels_placeholder: label_real # Discriminatorの出力ラベル(本物)
                        })

                    # 偽物画像を生成
                    image_fake = self.sess.run(self.output_gen_ops)

                    # 本物画像でDiscriminatorを学習
                    d_loss_real, d_acc_real = self.sess.run(self.train_disc_ops)

                    # Discriminatorに偽物画像を与えるため
                    # iteratorを初期化
                    self.sess.run(
                        self.iterator_init,
                        feed_dict={
                            self.images_placeholder: image_fake, # Discriminatorの入力(偽物)
                            self.noise_placeholder: noise, # Genratorの入力(使わないのでなんでもいい)
                            self.labels_placeholder: label_fake # Discriminatorの出力ラベル(偽物)
                        })

                    # 偽物画像でDiscriminatorを学習
                    d_loss_fake, d_acc_fake = self.sess.run(self.train_disc_ops)

                    # 本物画像の結果と偽物画像の結果を平均
                    d_loss = 0.5 * (d_loss_real + d_loss_fake)
                    d_acc = 0.5 * (d_acc_real + d_acc_fake)

                    #---------------------
                    # Generatorの学習
                    #---------------------

                    # iteratorを初期化
                    self.sess.run(
                        self.iterator_init,
                        feed_dict={
                            self.images_placeholder: image_real, # Discriminatorの入力(使わないのでなんでもいい)
                            self.noise_placeholder: noise, # Genratorの入力
                            self.labels_placeholder: label_real # Discriminatorの出力ラベル(本物)
                        })

                    # 本物ラベルでGeneratorを学習
                    g_loss, g_acc = self.sess.run(self.train_gen_ops)

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
                    noise = np.random.normal(0, 1, (self.batch_size, self.z_dim)).astype(np.float32)
                    self.sess.run(
                        self.iterator_init,
                        feed_dict={
                            self.images_placeholder: image_real, # Discriminatorの入力(使わないのでなんでもいい)
                            self.noise_placeholder: noise, # Genratorの入力
                            self.labels_placeholder: label_real # Discriminatorの出力ラベル(使わないのでなんでもいい)
                        })
                    image_fake = self.sess.run(self.output_gen_ops)
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

if __name__ == '__main__':
    G = GAN()
    G.fit()
