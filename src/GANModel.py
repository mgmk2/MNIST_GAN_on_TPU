import os
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential

class GANModel(object):
    def __init__(self, params, use_tpu=False, wgan_mode=False):
        self.params = params
        self.use_tpu = use_tpu
        self.wgan_mode = wgan_mode
        self.gp_weight = 10

        if self.use_tpu:
            # TPU対応のおまじない1
            tf.keras.backend.clear_session()
            tpu_grpc_url = "grpc://" + os.environ["COLAB_TPU_ADDR"]
            self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
            self.strategy = tf.contrib.distribute.TPUStrategy(self.tpu_cluster_resolver)
        self.build_model()

    def tpu_decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.use_tpu:
                with self.strategy.scope():
                    output = func(self, *args, **kwargs)
            else:
                output = func(self, *args, **kwargs)
            return output
        return wrapper

    def tpu_ops_decorator(mode):
        def _tpu_ops_decorator(func):
            def wrapper(self, *args, **kwargs):
                outputs = []
                if self.use_tpu:
                    _func = lambda *fargs, **fkwargs: func(self, *fargs, **fkwargs)
                    func_outputs = self.strategy.experimental_run_v2(_func, args=args, kwargs=kwargs)
                else:
                    func_outputs = func(self, *args, **kwargs)
                if not isinstance(func_outputs, tuple):
                    func_outputs = (func_outputs, )
                for x in func_outputs:
                    if isinstance(mode, str) and mode.upper() == 'SUM':
                        if self.use_tpu:
                            y = self.strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=None)
                        else:
                            y = tf.reduce_sum(x)
                    else:
                        if self.use_tpu:
                            y = tf.concat(x.values, axis=0)
                        else:
                            y = x
                    outputs.append(y)
                if len(outputs) > 1:
                    return tuple(outputs)
                elif len(outputs) == 1:
                    return outputs[0]
            return wrapper
        return _tpu_ops_decorator

    @tpu_decorator
    def build_model(self):
        # Discriminatorの定義
        self.discriminator = self.build_discriminator()
        self.optimizer_disc = tf.train.AdamOptimizer(2.0e-4, 0.5) # Discriminator用のOptimizer
        self.var_disc = self.discriminator.trainable_variables # Discriminatorの重み

        # Generatorの定義
        self.generator = self.build_generator()
        self.optimizer_gen = tf.train.AdamOptimizer(2.0e-4, 0.5) # Generator用のOptimizer
        self.var_gen = self.generator.trainable_variables # Generatorの重み

        # データセットの入力用placeholder
        self.images_placeholder = tf.placeholder(tf.float32, [None, *self.params.image_shape])
        self.noise_placeholder = tf.placeholder(tf.float32, [None, *self.params.noise_shape])

        # Dataset APIで入力パイプラインを定義
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.images_placeholder,
             self.noise_placeholder
            ))
        dataset = dataset.repeat()
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True) # TPUではdrop_remainder=Trueが必須

        # DatasetをTPU用のDatasetに変換
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)

        # iteratorを定義
        input_iterator = dist_dataset.make_initializable_iterator()
        self.iterator_init = input_iterator.initialize()

        # 学習等のopsを定義
        inputs = input_iterator.get_next() # ネットワークの入力
        self.output_gen_ops = self.output_images_gen(inputs) # Generatorの出力
        if self.wgan_mode:
            self.train_disc_ops = self.train_step_disc_W(inputs) # Discriminatorの学習
            self.train_gen_ops = self.train_step_gen_W(inputs) # Generatorの学習
        else:
            self.train_disc_real_ops = self.train_step_disc_real(inputs) # Discriminatorの学習
            self.train_disc_fake_ops = self.train_step_disc_fake(inputs) # Discriminatorの学習
            self.train_gen_ops = self.train_step_gen(inputs) # Generatorの学習

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

    def build_discriminator(self):
        # discriminatorモデル
        # kerasのSequentialを使っているが、Functional APIでもtensorflowの低レベルAPIでもたぶん大丈夫

        layers_disc = []
        layers_disc.append(
            Conv2D(16, (5, 5), strides=(2, 2), padding='same', input_shape=self.params.image_shape))
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
        layers_gen.append(Dense(7 * 7 * 256, use_bias=False, input_shape=self.params.noise_shape))
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

    @tpu_ops_decorator(mode=None)
    def output_images_gen(self, inputs):
        # Generatorの出力画像を得る
        _, noises = inputs # 入力データ
        return self.generator(noises, training=False) # GeneratorにBatchNormalizationを入れている場合はtraining=Falseを指定

    @tpu_ops_decorator(mode='SUM')
    def train_step_disc_real(self, inputs):
        # Discriminatorに対して
        # コストを計算して逆伝播法で重みを更新する

        features, _ = inputs # 入力データ
        labels = tf.ones(tf.stack([tf.shape(features)[0], 1]), tf.float32)
        logits = self.discriminator(features) # Discriminatorの出力

        # コスト関数と重み更新
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_sum(cross_entropy) / self.params.batch_size # reduce_meanは使わない方がいい
        train_op_disc = self.optimizer_disc.minimize(loss, var_list=self.var_disc) # discriminatorの重みのみ更新する

        # 精度
        logits_bool = tf.cast(tf.greater_equal(logits, 0), tf.float32)
        acc = tf.reduce_sum(1.0 - tf.abs(labels - logits_bool)) / self.params.batch_size

        # 必ずtf.control_dependenciesを使うこと
        with tf.control_dependencies([train_op_disc]):
            return tf.identity(loss), tf.identity(acc)

    @tpu_ops_decorator(mode='SUM')
    def train_step_disc_fake(self, inputs):
        # Discriminatorに対して
        # コストを計算して逆伝播法で重みを更新する

        _, noises = inputs # 入力データ
        features = self.generator(noises, training=False) # GeneratorにBatchNormalizationを入れている場合はtraining=Falseを指定
        labels = tf.zeros(tf.stack([tf.shape(features)[0], 1]), tf.float32)
        logits = self.discriminator(features) # Discriminatorの出力

        # コスト関数と重み更新
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_sum(cross_entropy) / self.params.batch_size # reduce_meanは使わない方がいい
        train_op_disc = self.optimizer_disc.minimize(loss, var_list=self.var_disc) # discriminatorの重みのみ更新する

        # 精度
        logits_bool = tf.cast(tf.greater_equal(logits, 0), tf.float32)
        acc = tf.reduce_sum(1.0 - tf.abs(labels - logits_bool)) / self.params.batch_size

        # 必ずtf.control_dependenciesを使うこと
        with tf.control_dependencies([train_op_disc]):
            return tf.identity(loss), tf.identity(acc)

    @tpu_ops_decorator(mode='SUM')
    def train_step_gen(self, inputs):
        # Generatorに対して
        # コストを計算して逆伝播法で重みを更新する

        _, noises = inputs # 入力データ
        features = self.generator(noises, training=True) # GeneratorにBatchNormalizationを入れている場合はtraining=Trueを指定
        logits = self.discriminator(features)
        labels = tf.ones(tf.stack([tf.shape(logits)[0], 1]), tf.float32)

        # コスト関数と重み更新
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_sum(cross_entropy) / self.params.batch_size
        train_op_gen = self.optimizer_gen.minimize(loss, var_list=self.var_gen) # Generatorの重みのみ更新

        # 精度
        logits_bool = tf.cast(tf.greater_equal(logits, 0), tf.float32)
        acc = tf.reduce_sum(1.0 - tf.abs(labels - logits_bool)) / self.params.batch_size

        # BatchNormalizationの平均と分散の更新
        # GeneratorにBatchNormalizationを入れている場合は必須
        update_ops = self.generator.get_updates_for(None) + self.generator.get_updates_for(noises)

        # 必ずtf.control_dependenciesを使うこと
        # BatchNormalizationを使っている場合はupdate_opsも一緒に入れる
        with tf.control_dependencies([train_op_gen, *update_ops]):
            return tf.identity(loss), tf.identity(acc)

    @tpu_ops_decorator(mode='SUM')
    def train_step_disc_W(self, inputs):
        # Discriminatorに対して
        # コストを計算して逆伝播法で重みを更新する

        features_real, noises = inputs # 入力データ
        features_fake = self.generator(noises, training=False) # GeneratorにBatchNormalizationを入れている場合はtraining=Falseを指定
        ratio = tf.random.uniform(tf.stack([tf.shape(features_real)[0], 1, 1, 1]), dtype=tf.float32)
        features_mix = ratio * features_real + (1.0 - ratio) * features_fake

        logits_real = self.discriminator(features_real) # Discriminatorの出力
        logits_fake = self.discriminator(features_fake) # Discriminatorの出力
        logits_mix = self.discriminator(features_mix) # Discriminatorの出力

        grad_mix, = tf.gradients(ys=logits_mix, xs=features_mix)
        norm_grad_mix = tf.sqrt(tf.reduce_sum(grad_mix ** 2, axis=[1, 2, 3]))
        grad_penalty = tf.reduce_sum((1.0 - norm_grad_mix) ** 2) / self.params.batch_size

        # コスト関数と重み更新
        loss = tf.reduce_sum(logits_fake - logits_real) / self.params.batch_size
        loss += self.gp_weight * grad_penalty
        train_op_disc = self.optimizer_disc.minimize(loss, var_list=self.var_disc) # discriminatorの重みのみ更新する

        # 精度
        logits_real_bool = tf.cast(tf.greater_equal(logits_real, 0), tf.float32)
        logits_fake_bool = tf.cast(tf.greater_equal(logits_fake, 0), tf.float32)
        acc_real = tf.reduce_sum(1.0 - tf.abs(1 - logits_real_bool)) / self.params.batch_size
        acc_fake = tf.reduce_sum(1.0 - tf.abs(logits_fake_bool)) / self.params.batch_size
        acc = 0.5 * (acc_real + acc_fake)

        # 必ずtf.control_dependenciesを使うこと
        with tf.control_dependencies([train_op_disc]):
            return tf.identity(loss), tf.identity(acc)

    @tpu_ops_decorator(mode='SUM')
    def train_step_gen_W(self, inputs):
        # Generatorに対して
        # コストを計算して逆伝播法で重みを更新する

        _, noises = inputs # 入力データ
        features = self.generator(noises, training=True) # GeneratorにBatchNormalizationを入れている場合はtraining=Trueを指定
        logits = self.discriminator(features)

        # コスト関数と重み更新
        loss = tf.reduce_sum(-logits) / self.params.batch_size
        train_op_gen = self.optimizer_gen.minimize(loss, var_list=self.var_gen) # Generatorの重みのみ更新

        # 精度
        logits_bool = tf.cast(tf.greater_equal(logits, 0), tf.float32)
        acc = tf.reduce_sum(1.0 - tf.abs(labels - logits_bool)) / self.params.batch_size

        # BatchNormalizationの平均と分散の更新
        # GeneratorにBatchNormalizationを入れている場合は必須
        update_ops = self.generator.get_updates_for(None) + self.generator.get_updates_for(noises)

        # 必ずtf.control_dependenciesを使うこと
        # BatchNormalizationを使っている場合はupdate_opsも一緒に入れる
        with tf.control_dependencies([train_op_gen, *update_ops]):
            return tf.identity(loss), tf.identity(acc)
