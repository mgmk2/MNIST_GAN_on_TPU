class Params(object):
    def __init__(self):
        self.z_dim = 100 # 潜在変数の次元

        self.image_shape = (28, 28, 1) # 画像のサイズ
        self.noise_shape = (self.z_dim,) # ノイズのサイズ

        self.epochs = 1000 # 学習回数
        self.batch_size = 512 # バッチサイズ
