import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--path',
                        type=str,
                        default=r"D:\Personals\GANS\colorization-main\images",
                        help="the train image fold")
    parser.add_argument('-s',
                        '--save_path',
                        type=str,
                        default=r'D:\Personals\GANS\colorization-main\practice\save_model')
    parser.add_argument('-l',
                        '--lr',
                        type=float,
                        default=0.0002,
                        help='initial learning rate')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=200,
                        help='the number of epochs in training process')

    parser.add_argument('--perceptual_weight',
                        type=float,
                        default=0.2,
                        help='weight of the perceptual loss')
    # gan parameters
    parser.add_argument('--zdim',
                        type=int,
                        default=64,
                        help='dimension of the generated noise for Generator')
    parser.add_argument('--adv_D_loss_weight',
                        type=float,
                        default=1.,
                        help='weight of adversarial loss for Discriminator')
    parser.add_argument('--gp_loss_weight',
                        type=float,
                        default=10,
                        help='weight of gradient penalty loss')
    parser.add_argument('--adv_G_loss_weight',
                        type=float,
                        default=0.001,
                        help='weight of adversarial loss for Generator')
    parser.add_argument('--pixel_loss_weight',
                        type=float,
                        default=2.,
                        help='weight of pixelwise loss for Generator')
    parser.add_argument('--perceptual_loss_weight',
                        type=float,
                        default=0.2,
                        help='weight of perceptual loss for Generator')

    parser.add_argument('--val_ratio',
                        type=float,
                        default=0.1,
                        help='validate ratio')
    parser.add_argument('--perceptual_loss',
                        type=bool,
                        default=True,
                        help='whether to use perceptual loss or not')
    parser.add_argument('--iter_print',
                        type=int,
                        default=100,
                        help='the iterations to print the training info')
    parser.add_argument('--valid_epoch',
                        type=int,
                        default=2,
                        help='check and save model every \"valid_epoch\" epochs')
    args = parser.parse_args()
    return args
