import os
import collections

import torch
import torchvision.transforms as T

from PIL import Image, ImageOps
import numpy as np
from statistics import mean

from basic_model import Net
from model import Generator
from config import load_config
from evaluation import getSSIM, getPSNR     # getIQM,

if __name__ == '__main__':
    # configuration
    args = load_config()
    if torch.cuda.is_available() and args.cuda is True:
        cuda = True
    else:
        cuda = False

    # model
    # In case we may have a mistake to choose a mode
    print('current mode: ', args.mode)
    if args.mode == 'basic':
        model = Net()
    else:
        model = Generator(args)

    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # data path
    path = args.path
    if os.path.isdir(path):
        ids = os.listdir(path)
        img_names = [os.path.join(path, img_id) for img_id in ids]
        args.show_result = False
        args.save_generated_result = True
    else:
        img_names = [path]

    # inference
    input_transform = T.Compose([T.ToPILImage(),
                                 T.Resize(size=(256, 256)),
                                 T.Grayscale(),
                                 T.ToTensor(),
                                 T.Normalize(0.5, 0.5)
                                 ])
    if args.mode != 'basic':
        input_transform_128 = T.Compose([T.ToPILImage(),
                                         T.Resize(size=(128, 128)),
                                         T.Grayscale(),
                                         T.ToTensor(),
                                         T.Normalize(0.5, 0.5)
                                         ])
        input_transform_64 = T.Compose([T.ToPILImage(),
                                        T.Resize(size=(64, 64)),
                                        T.Grayscale(),
                                        T.ToTensor(),
                                        T.Normalize(0.5, 0.5)
                                        ])
    ms = collections.defaultdict(dict)
    for img_name in img_names:
        print('image name: ', img_name)

        img_id = img_name.split('/')[-1]
        original_image = Image.open(img_name)
        target_image = original_image.resize((256, 256))
        gray_image = ImageOps.grayscale(target_image)
        np_image = np.asarray(original_image)
        # print('np image shape: ', np_image.shape)
        input_image = input_transform(np_image).unsqueeze(0)
        # print('input_image shape: ', input_image.shape)

        if args.mode != 'basic':
            np_image_128 = np.asarray(original_image)
            np_image_64 = np.asarray(original_image)
            input_image_128 = input_transform_128(np_image_128).unsqueeze(0)
            input_image_64 = input_transform_64(np_image_64).unsqueeze(0)
            if args.mode == 'gan':
                z = torch.rand((1, 1, 8, 8))
                if args.cuda:
                    z = z.cuda()
            else:
                z = 0

        model.eval()
        if cuda:
            input_image = input_image.cuda()
            if args.mode != 'basic':
                input_image_128 = input_image_128.cuda()
                input_image_64 = input_image_64.cuda()

        with torch.no_grad():
            if args.mode == 'basic':
                pred_img = model(input_image)
            else:
                pred_img = model(input_image, input_image_128, input_image_64, z)
            pred_img = pred_img.squeeze(0).permute(1, 2, 0).cpu()
            # print('pred image shape: ', pred_img.shape)

        pred_min, pred_max = pred_img.min(), pred_img.max()
        pred_img.clamp_(min=pred_min, max=pred_max)
        pred_img.add_(-pred_min).div_(pred_max - pred_min)
        color_image = pred_img.mul(255).clamp(0, 255).byte().numpy()
        color_image = Image.fromarray(np.uint8(color_image))

        if args.show_result:
            original_image.show()
            target_image.show()
            gray_image.show()
            color_image.show()

        if args.save_generated_result:
            os.makedirs(args.save_generated_result_path, exist_ok=True)
            # save generated results
            color_image_name = os.path.join(args.save_generated_result_path, img_id.split('.')[0] + '_color.jpg')
            color_image.save(color_image_name)
            # save corresponding source image (resized to 256 x 256)
            source_image_name = os.path.join(args.save_generated_result_path, img_id.split('.')[0] + '_ori.jpg')
            target_image.save(source_image_name)
            # save corresponding gray image to the source image
            gray_image_name = os.path.join(args.save_generated_result_path, img_id.split('.')[0] + '_gray.jpg')
            gray_image.save(gray_image_name)

        if args.do_eval:
            # ground truth: target_image | colorized: color_image
            psnr = getPSNR(np.array(target_image), np.array(color_image))
            ssim = getSSIM(np.array(target_image), np.array(color_image))
            ms['spnr'][img_name] = psnr
            ms['ssim'][img_name] = ssim

    if args.do_eval:
        mean_spnr = mean(list(ms['spnr'].values()))
        mean_ssim = mean(list(ms['ssim'].values()))
        ms['spnr']['mean'] = mean_spnr
        ms['ssim']['mean'] = mean_ssim
        print('mean spnr: ', mean_spnr)
        print('mean ssim: ', mean_ssim)

        if args.save_eval:
            os.makedirs(args.save_eval_path, exist_ok=True)
            np.save(os.path.join(args.save_eval_path, 'eval.npy'), ms)




