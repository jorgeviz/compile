import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, default="output",
                    dest="log_dir",
                    help='Logging directory.')
parser.add_argument('--off-screen', action='store_true', default=False,
                    help='Plot on screen.')
args = parser.parse_args()

def read_files(pdir):
    data = []
    for _f in os.listdir(pdir):
        data.append(
            (_f.split('.')[0], pd.read_csv(os.path.join(pdir, _f)) )
        )
    return data

def add_plot(df, axs, file_name):
    # NLL
    axs[0, 0].plot(df['step'], df['nll']) 
    axs[0, 0].set_title('NLL')
    # axs[0, 0].set(ylabel='loss')
    # Reconstruction Acc
    axs[0, 1].plot(df['step'], df['rec_acc']) 
    axs[0, 1].set_title('Reconstruction')
    # axs[0, 1].set(ylabel='accuracy')
    # Batch NLL
    axs[1, 0].plot(df['step'], df['batch_nll']) 
    axs[1, 0].set_title('Batch NLL')
    # axs[1, 0].set(ylabel='loss')
    # Batch Reconstruction Acc
    axs[1, 1].plot(df['step'], df['batch_rec_acc'], label=file_name) 
    axs[1, 1].set_title('Batch Reconstruction')
    axs[1, 1].legend(loc="lower right")
    # axs[1, 1].set(ylabel='accuracy')

    for ax in axs.flat:
        ax.set(xlabel='steps')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        

if __name__ == "__main__":
    plt_data = read_files(args.log_dir)
    # Initialize plot
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('CompILE')
    # Add plots for each file
    for k, df in plt_data:
        add_plot(df, axs, k)
    # Display or save
    if not args.off_screen:
        plt.show()
    else:
        plt.savefig(args.log_dir+"/CompILE_benchmark.png")