import argparse


def load_config():
    parser = argparse.ArgumentParser()
    # dataset & checkpoint
    parser.add_argument('-d',
                        '--path',
                        type=str,
                        # default='D:/my_doc/inter/interview/neon/landscape_images',
                        # default = 'D:/my_doc/inter/interview/neon/landscape_images/958.jpg',
                        # default='/home/liangkezhao/workspace/neon/landscape_images',
                        default='/home/liangkezhao/workspace/neon/landscape_images/666.jpg',
                        help='data path: image folder path for training, and image path when inference')
    parser.add_argument('-s',
                        '--save_path',
                        # default='D:/my_doc/inter/interview/neon/pths'
                        default='/home/liangkezhao/workspace/neon/pths',
                        help='location to save models')

    # training parameters
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=200,
                        help='default training epochs')
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
    parser.add_argument('--milestones',
                        default=[30, 50],
                        metavar='N',
                        nargs='*',
                        help='epochs at which learning rate is timed by gamma'
                        )
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help='learning rate is timed by gamma when meets a milestone'
                        )
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
                        default=10.,
                        help='weight of gradient penalty loss')
    parser.add_argument('--adv_G_loss_weight',
                        type=float,
                        default=0.001,
                        help='weight of adversarial loss for Generator')
    parser.add_argument('--pixel_loss_weight',
                        type=float,
                        default=1.,
                        help='weight of pixelwise loss for Generator')
    parser.add_argument('--perceptual_loss_weight',
                        type=float,
                        default=0.2,
                        help='weight of perceptual loss for Generator')

    # training pipeline
    parser.add_argument('-m',
                        '--mode',
                        default='gan',
                        choices=['basic', 'advanced', 'gan'],
                        help='running mode of this project. \n'
                             'basic: use the provided simple network. \n'
                             'advanced: use the FPN-like advanced network. \n'
                             'gan: use gan based method')
    parser.add_argument('-r',
                        '--reload',
                        type=bool,
                        default=False,
                        help='if True, reload existing model and finetune.\n'
                             '         the name of the model is defined by --resume\n'
                             'if False, train from scratch')
    parser.add_argument('--solver',
                        type=str,
                        default='adam',
                        choices=['adam', 'sgd'],
                        help='choose your optimizer. only adam & sgd is supported now')
    parser.add_argument('--valid_epoch',
                        type=int,
                        default=1,
                        help='check and save model every \"valid_epoch\" epochs')
    parser.add_argument('--show_result',
                        type=bool,
                        default=True,
                        help='show colored result using the trained model')
    parser.add_argument('--resume',
                        type=str,
                        # default='D:/my_doc/inter/interview/neon/pths/gan/G_gan_best.pth',
                        default='/home/liangkezhao/workspace/neon/pths/gan/G_gan_best.pth',
                        # default='/home/liangkezhao/workspace/neon/pths/basic/basic_best.pth',
                        # default='/home/liangkezhao/workspace/neon/pths/advanced/advanced_best.pth',
                        help='trained model. used for retraining')
    parser.add_argument('--save_generated_result',
                        type=bool,
                        default=True,
                        help='whether or not save the generated results')
    parser.add_argument('--save_generated_result_path',
                        type=str,
                        # default='D:/my_doc/inter/interview/neon/generated'
                        default='/home/liangkezhao/workspace/neon/generated',
                        help='path of the saving generated results')

    # Evaluation
    parser.add_argument('--do_eval',
                        type=bool,
                        default=True,
                        help='do evaluation simultaneously when doing inference')
    parser.add_argument('--save_eval',
                        type=bool,
                        default=False,
                        help='whether to save evaluation result')
    parser.add_argument('--save_eval_path',
                        type=str,
                        # default='D:/my_doc/inter/interview/neon/eval'
                        default='/home/liangkezhao/workspace/neon/eval',
                        help='save the evaluation result')

    # miscellaneous: may not need to change
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='seed for randomly generating dataset')
    parser.add_argument('--cuda',
                        type=bool,
                        default=True,
                        help='gpu or cpu')
    parser.add_argument('--val_ratio',
                        type=float,
                        default=0.1,
                        help='validate ratio')
    parser.add_argument('--perceptual_loss',
                        type=bool,
                        default=True,
                        help='whether to use perceptual loss or not')
    parser.add_argument('--use_batchnorm',
                        type=bool,
                        default=True,
                        help='whether to use batch normalization or not')
    parser.add_argument('--wandb',
                        type=bool,
                        default=False,
                        help='check training/validating result online. '
                             'It\'s not free so set to \"False\" when inference')

    args = parser.parse_args()

    return args
