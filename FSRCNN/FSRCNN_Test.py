"""
IN THE NAME OF GOD

Subject: Converts LR images to HR images

Created on Sun Jan 23 09:12:28 2022

@Author: Pouya Sadeghi
"""

#====================================================================================================================================================================================
"""
SECTION 1 : Import Necessary Packages (Libraries)
"""
from datasets import Image_Dataset
from utils import calc_psnr
from models import FSRCNN
import numpy as np
import h5py
import copy
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import time
import torch; torch.manual_seed(0)
import torch.distributions
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('grayscale')
# mpl.style.use('classic')
# from skimage.metrics import structural_similarity
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.max_open_warning': 0})
#==========================================================================================================================================
if __name__ == '__main__':
    inp = {
        'db_path':          "C:\\Users/Mehrgan/PycharmProjects/Nokhbeh/Finalist/HR_Finalist.h5",
        'db_path_new': "C:\\Users/Mehrgan/PycharmProjects/Nokhbeh/Finalist2/LR_Finalist.h5",
        'LRsample_shape': (1, 32, 32),  # (Channel, Height, Width)
        'HRsample_shape': (1, 128, 128),  # (Channel, Height, Width)
        'outputs-dir': 'C:\\Users/Mehrgan/PycharmProjects/Article 2/Asli/FSRCNN2/output/',
        'scale': 4,
        'num_epochs': 100,
        'num_workers': 8,
        'seed': 123,
        'optimizer': 'SGD',
        'trDB_size': -1,  # -1 to use all Images available in the db
    }

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device:  {device}')

    t = time.time()

    model = FSRCNN(scale_factor=inp['scale']).to(device)  # GPU
    # model = FSRCNN().to(device)  # GPU
    summary(model, inp['LRsample_shape'], device="cuda")
    print(model)
    inp['model'] = str(model)

    from colorama import Fore

    print(Fore.YELLOW + """" **************************************** Loading dataset **************************************** """)
    # pouya = 2675
    with h5py.File(inp['db_path'], "r") as hf:
        # Split the Data into Training / Test
        # Split the Data into Features / Targets
        train_y = np.array(hf['3D_image'][:inp['trDB_size']])

    Test_y = train_y[15000:18000]
    # Test_y = train_y[14999+pouya:15000+pouya]
    # =============================================================================
    with h5py.File(inp['db_path_new'], "r") as hf:
        # Split the Data into Training / Test
        # Split the Data into Features / Targets
        train_x = np.array(hf['3D_image'][:inp['trDB_size']])

    Test_x = train_x[15000:18000]
    # Test_x = train_x[14999+pouya:15000+pouya]

    Test_Dataset = Image_Dataset(Test_x, Test_y, inp['LRsample_shape'], inp['HRsample_shape'], device)
    print('TestDB:', Test_Dataset.img_x.shape[0])
    Test_dataloader = DataLoader(dataset=Test_Dataset,
                                 batch_size=1)
    #============================================================================

    print(Fore.GREEN + """" **************************************** Start of Testing Network **************************************** """)

#===================================================================================================================================================================================
"""
SECTION 5 : Test Model
"""
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
chk_Path = f'./Image Output/'
# from skimage.metrics import structural_similarity as ssim
import lpips
import pandas as pd

#============================================================
def plot_imgs_test(inp, outp, gt, epoch):
    # input image, reconstructed image, ground truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(f'Sample = {epoch}')
    ax1.imshow(inp.cpu().detach().numpy()[0, 0, :, :])
    ax1.set_title('Input')
    ax2.imshow(outp.cpu().detach().numpy()[0, 0, :, :])
    ax2.set_title('Reconstructed')
    ax3.imshow(gt.cpu().detach().numpy()[0, 0, :, :])
    ax3.set_title('Ground truth')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{chk_Path}/{epoch}.png', dpi=300)
    time.sleep(0.1)
    plt.close(fig)

def plot_imgs_fsrcnn(inp, epoch):
    plt.imshow(inp[0][0])
    plt.tight_layout()
    plt.savefig(f'{epoch}_FSRCNN.png', dpi=300)
    time.sleep(0.1)

def plot_imgs_lr(inp, epoch):
    plt.imshow(inp[0][0])
    plt.tight_layout()
    plt.savefig(f'{epoch}_LR.png', dpi=300)
    time.sleep(0.1)

def plot_imgs_hr(inp, epoch):
    plt.imshow(inp[0][0])
    plt.tight_layout()
    plt.savefig(f'{epoch}_HR.png', dpi=300)
    time.sleep(0.1)
#============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args(['--weights-file', 'C:\\Users/Mehrgan/PycharmProjects/Article 2/Asli/FSRCNN2/output/best.pth',
                                            '--image-file', 'C:\\Users/Mehrgan/PycharmProjects/Article 2/FSRCNN2/3/Image Folder/'])

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    i = 0

    model.eval()
    for x, y in Test_dataloader:
        with torch.no_grad():
            sr = model(x).clamp(0.0, 1.0)
            PSNR = calc_psnr(y, sr)
            print(PSNR)
            # sr = sr.cpu().numpy()
            # y = y.cpu()
            # y = y.numpy()
            # SSIM = ssim(sr[0][0], y[0][0], channel_axis=1)
            # print(SSIM)
            # matplotlib.image.imsave(f'C:\\Users/Mehrgan/PycharmProjects/Article 2/Asli/FSRCNN2/Images/{i}_FSRCNN.png', sr[0][0])
            # plot_imgs_fsrcnn(sr, i)

            i+=1
#=================================================================================================================
"""
END
"""
#=================================================================================================================