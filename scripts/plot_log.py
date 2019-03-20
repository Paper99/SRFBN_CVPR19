import argparse

import matplotlib.pyplot as plt
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Plot Logs')
    parser.add_argument('-path', type=str, required=True, help="Path to '.csv' file.")

    df = pd.read_csv(parser.parse_args().path)

    epoch = df['epoch']
    train_loss = df['train_loss']
    val_loss = df['val_loss']
    psnr = df['psnr']
    ssim = df['ssim']

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))

    axs[0].plot(epoch, train_loss, label='train_loss')
    axs[0].plot(epoch, val_loss, label='val_loss')
    axs[0].set(xlabel='Epoch', ylabel='Loss',
               title='Train (Val) Loss Analysis')
    axs[0].legend()

    axs[1].plot(epoch, psnr)
    axs[1].set(xlabel='Epoch', ylabel='PSNR (dB)',
               title='PSNR Analysis')

    axs[2].plot(epoch, ssim)
    axs[2].set(xlabel='Epoch', ylabel='SSIM',
               title='SSIM Analysis')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
