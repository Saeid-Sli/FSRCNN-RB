"""
IN THE NAME OF GOD

Subject: Converts LR images to HR images

@Author: Pouya Sadeghi
"""
#=================================================================================================================================================================================
"""
SECTION : Import Necessary Packages (Libraries)
"""
from datasets import Image_Dataset
from utils import calc_psnr, AverageMeter, plot_loss_animation, plot_imgs, save_multi_image
from models import FSRCNN
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import copy
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time
from datetime import timedelta, datetime
import pandas as pd
import torch; torch.manual_seed(0)
import torch.distributions
from torchsummary import summary
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
# mpl.style.use('grayscale')
mpl.style.use('classic')
from tabulate import tabulate
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.max_open_warning': 0})
#==================================================================================================================================================================================
"""
SECTION: Train Model
"""
if __name__ == '__main__':
    inp = {
        'db_path':          "C:\\Users/Mehrgan/PycharmProjects/Nokhbeh/Finalist/HR_Finalist.h5",
        'db_path_new': "C:\\Users/Mehrgan/PycharmProjects/Nokhbeh/Finalist2/LR_Finalist.h5",
        'LRsample_shape': (1, 32, 32),  # (Channel, Height, Width)
        'HRsample_shape': (1, 128, 128),  # (Channel, Height, Width)
        'outputs-dir': 'E:\\FSRCNN/New folder/',
        'scale': 4,
        'learning_rate': 1e-3,
        'batch_size': 15,
        'num_epochs': 100,
        'num_workers': 8,
        'seed': 123,
        'optimizer': 'Adam',
        'trDB_size': -1,  # -1 to use all Images available in the db
    }

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device:  {device}')

    torch.manual_seed(inp['seed'])

    # For Test/Continue_Train chkPath_load at below should be set too.
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------
    DatafolderName = os.path.basename(inp['db_path'])
    chkp_Path = f'./_results/8.{DatafolderName[:-3]}_{inp["optimizer"]}_b{inp["batch_size"]}_lr{inp["learning_rate"]}_LR2HR-model3'
    pathlib.Path(chkp_Path).mkdir(parents=True, exist_ok=True)

    t = time.time()

    model = FSRCNN(scale_factor=inp['scale']).to(device)  # GPU
    # model = FSRCNN().to(device)  # GPU
    summary(model, inp['LRsample_shape'], device="cuda")
    print(model)
    inp['model'] = str(model)
    criterion = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'learning_rate': inp['learning_rate'] * 0.1}
    ], lr=inp['learning_rate'])
    # {'params': model.last_part.parameters()}])

    from colorama import Fore

    print(Fore.YELLOW + """" **************************************** Loading dataset **************************************** """)

    with h5py.File(inp['db_path'], "r") as hf:
        # Split the Data into Training / Test
        # Split the Data into Features / Targets
        # train_x = np.array(hf["LR/Train"]['3D_image'][:inp['trDB_size']])
        train_y = np.array(hf['3D_image'][:inp['trDB_size']])

    Train_y = train_y[:10000]
    Eval_y = train_y[10000:11800]
    Test_y = train_y[17500:]

    # =============================================================================
    with h5py.File(inp['db_path_new'], "r") as hf:
        # Split the Data into Training / Test
        # Split the Data into Features / Targets
        train_x = np.array(hf['3D_image'][:inp['trDB_size']])

    Train_x = train_x[:10000]
    Eval_x = train_x[10000:11800]
    Test_x = train_x[17500:]

    Train_Dataset = Image_Dataset(Train_x, Train_y, inp['LRsample_shape'], inp['HRsample_shape'], device)
    print('TrainDB:', Train_Dataset.img_x.shape[0])
    train_dataloader = DataLoader(dataset=Train_Dataset,
                                  batch_size=inp['batch_size'],
                                  shuffle=True)
    # num_workers = inp['num_workers'],
    # pin_memory = True)

    Eval_Dataset = Image_Dataset(Eval_x, Eval_y, inp['LRsample_shape'], inp['HRsample_shape'], device)
    print('TrainDB:', Eval_Dataset.img_x.shape[0])
    Eval_dataloader = DataLoader(dataset=Eval_Dataset,
                                 batch_size=1)

    Test_Dataset = Image_Dataset(Test_x, Test_y, inp['LRsample_shape'], inp['HRsample_shape'], device)
    print('TestDB:', Test_Dataset.img_x.shape[0])
    Test_dataloader = DataLoader(dataset=Test_Dataset,
                                 batch_size=1)

    # ============================================================================
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    elapsed = time.time() - t
    print('Dataset loading time=', str(timedelta(seconds=elapsed)))

    start_epoch = 0

    print(Fore.GREEN + """" **************************************** Start of Training Network **************************************** """)
    iteration_list, loss_list = [], []
    fig_params = None
    loss_avg = 0
    Epoch = np.arange(0, 1000)
    Loss_Training = np.zeros(1000)
    PSNR = np.zeros(1000)
    Epoch_elapsed_time = np.zeros(1000)
    t_total = time.time()

    for epoch in range(start_epoch, inp['num_epochs']):
        t_epoch = time.time()
        model.train()
        epoch_losses = AverageMeter()

        print(Fore.GREEN + '\nThe Information of epoch {} as Follows: '.format(epoch))
        with tqdm(total=(len(Train_Dataset) - len(Train_Dataset) % inp['batch_size']), position=0, leave=True) as t:
            t.set_description('epoch: {}/{}'.format(epoch, inp['num_epochs'] - 1))

            for x, y in train_dataloader:
                optimizer.zero_grad()
                # attention_matrix = torch.randint(self._initialize_weights.size())
                # model.fc.weight = model.fc.weight * attention_matrix
                y_preds = model (x)
                criterion = nn.MSELoss()
                loss = criterion(y_preds, y)
                loss_avg += loss.item()
                loss.backward()
                optimizer.step()
                epoch_losses.update(loss.item(), len(x))

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(x))
            # scheduler.step()
            torch.save(model.state_dict(), os.path.join(inp['outputs-dir'], 'epoch_{}.pth'.format(epoch)))
            loss_avg /= len(train_dataloader)
            loss_list.append(loss_avg)
            iteration_list.append(epoch)

            # Plot loss animaiton
            fig_params = plot_loss_animation(iteration_list, loss_list, fig_params)

            # plot reconstructed
            if epoch % 10 == 0:
                plot_imgs(x, y_preds, y, epoch)

            # Print Loss
            # print(f'Epoch: {epoch}, Loss_Training: {loss_avg:.8f}')

            # print(tabulate([['epoch', epoch], ['loss_avg', loss_avg]], headers=['Epoch', 'Loss_Training']))
            print(tabulate([["\nEpoch", "\nLoss_Training"], ["epoch", epoch], ["loss_avg", loss_avg]], headers = "firstrow"))
            Loss_Training[epoch] = loss_avg
            # print(f'min: {y_preds.min().item()} max: {y_preds.max().item()}')
            elapsed = time.time() - t_epoch
            print('Epoch elapsed time=', str(timedelta(seconds=elapsed)))
            Epoch_elapsed_time[epoch] = elapsed

            model.eval()
            epoch_psnr = AverageMeter()
            for x, y in Eval_dataloader:
                with torch.no_grad():
                    preds = model(x).clamp(0.0, 1.0)
                    # preds_SSIM = model(x)
                    # score = structural_similarity(preds_SSIM, y)

                epoch_psnr.update(calc_psnr(preds, y), len(x))

            # print('Eval PSNR: {:.2f}'.format(epoch_psnr.avg))
            table = [["PSNR", epoch_psnr.avg]]
            print(tabulate(table, headers=["\nPeak Signal-to-Noise ratio", "\nValue of PSNR"]))
            # print(tabulate([["\nPSNR"], ["Eval PSNR", epoch_psnr.avg], ["loss_avg", loss_avg]], headers = "firstrow"))

            print(Fore.BLUE + '==================================================================================================================================')
            PSNR[epoch] = epoch_psnr.avg
            # print("Image similarity: {}".format(score))

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

    print(Fore.MAGENTA + 'Best Epoch: {}'.format(best_epoch))
    print(Fore.YELLOW + '==========================================')
    print(Fore.RED + 'Best PSNR: {}'.format(best_psnr))
    print(Fore.YELLOW + '==========================================')
    torch.save(best_weights, os.path.join(inp['outputs-dir'], 'best.pth'))
    # Save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'{chkp_Path}/epoch{epoch}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pt')
    # save loss vs epoch image
    # plt.savefig(f'{chkp_Path}/loss.png')
    # np.savetxt("PSNR.csv", PSNR, delimiter=",", header='PSNR')
    # np.savetxt("Loss_Training.csv", Loss_Training, delimiter=",", header='Loss_Training')
    df = pd.DataFrame({"Epoch": Epoch, "Loss_Training": Loss_Training, "PSNR": PSNR, "Epoch_elapsed_time": Epoch_elapsed_time})
    df.to_csv("output.csv", index=False)

    save_multi_image(f'{chkp_Path}/loss.pdf')
    time_training_network = time.time() - t_total
    print(f'The Time of Train Model: {time_training_network}')
#=================================================================================================================================================================================
"""
END
"""
#==================================================================================================================================================================================