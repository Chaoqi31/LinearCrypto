import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import yaml
import torch
import matplotlib
import numpy as np
from utils import io_tools
from datetime import datetime
import pytorch_lightning as pl
import matplotlib.ticker as ticker
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
sns.set_theme(style='whitegrid', context='talk', font_scale=1.5)
palette = sns.color_palette('bright')



ROOT = io_tools.get_root(__file__, num_returns=2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging directory.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of computing devices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Logging directory.",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default='Cmamba',
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='cmamba_nv',
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        '--use_volume', 
        default=False,   
        action='store_true',          
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )

    args = parser.parse_args()
    return args

def print_and_write(file, txt, add_new_line=True):
    print(txt)
    if add_new_line:
        file.write(f'{txt}\n')
    else:
        file.write(txt)

def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    save_dict.pop('checkpoint_callback')
    with open(log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)

def init_dirs(args, name):
    # Extract version directory from checkpoint path
    # e.g., logs/CMamba_Binance/version_1/checkpoints/last.ckpt -> logs/CMamba_Binance/version_1
    ckpt_path = args.ckpt_path
    if 'checkpoints' in ckpt_path:
        version_dir = os.path.dirname(os.path.dirname(ckpt_path))
    else:
        version_dir = os.path.dirname(ckpt_path)

    # Create results subdirectory in the version folder
    path = f'{version_dir}/results'
    if not os.path.isdir(path):
        os.makedirs(path)
    print(f'Saving results to: {path}')

    txt_file = open(f'{path}/metrics.txt', 'w')
    plot_path = f'{path}/pred.png'
    return txt_file, plot_path

def load_model(config, ckpt_path):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get('normalize', False)
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'))
    model.cuda()
    model.eval()
    return model, normalize

@torch.no_grad()
def run_model(model, dataloader, factors=None):
    target_list = []
    preds_list = []
    timetamps = []
    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get('Timestamp').numpy().reshape(-1)
            target = batch.get(model.y_key).numpy().reshape(-1)
            features = batch.get('features').to(model.device)
            preds = model(features).cpu().numpy().reshape(-1)
            target_list += [float(x) for x in list(target)]
            preds_list += [float(x) for x in list(preds)]
            timetamps += [float(x) for x in list(ts)]

    if factors is not None:
        scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
        shift = factors.get(model.y_key).get('min')
        target_list = [x * scale + shift for x in target_list]
        preds_list = [x * scale + shift for x in preds_list]
        scale = factors.get('Timestamp').get('max') - factors.get('Timestamp').get('min')
        shift = factors.get('Timestamp').get('min')
        timetamps = [x * scale + shift for x in timetamps]
    targets = np.asarray(target_list)
    preds = np.asarray(preds_list)
    targets_tensor = torch.tensor(target_list)
    preds_tensor = torch.tensor(preds_list)
    timetamps = [datetime.fromtimestamp(int(x)) for x in timetamps]
    mse = float(model.mse(preds_tensor, targets_tensor))
    mape = float(model.mape(preds_tensor, targets_tensor))
    l1 = float(model.l1(preds_tensor, targets_tensor))
    return timetamps, targets, preds, mse, mape, l1



if __name__ == "__main__":

    args = get_args()
    pl.seed_everything(args.seed)
    logdir = args.logdir

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')
    name = config.get('name', args.expname)

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")

    use_volume = args.use_volume
    if not use_volume:
        use_volume = config.get('use_volume')
    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))

    model, normalize = load_model(config, args.ckpt_path)
    data_module = CMambaDataModule(data_config,
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform,
                                   batch_size=args.batch_size,
                                   distributed_sampler=False,
                                   num_workers=args.num_workers,
                                   normalize=normalize,
                                   )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    dataloader_list = [train_loader, val_loader, test_loader]
    titles = ['Train', 'Val', 'Test']
    split_colors = {
        'Train': '#ff7f0e',   # bright orange
        'Val':   '#00c853',   # vivid green
        'Test':  '#ff2d55',   # vivid pink
        'Target': '#00c4ff',  # bright cyan
    }
    line_styles = {
        'Train': '--',
        'Val': '--',
        'Test': '--',
        'Target': '-',
    }

    factors = None
    if normalize:
        factors = data_module.factors
    all_targets = []
    all_timestamps = []

    # Store test data for separate plot
    test_timestamps = None
    test_targets = None
    test_preds = None


    f, plot_path = init_dirs(args, name)

    fig, ax = plt.subplots(figsize=(24, 12))
    print_format = '{:^7} {:^15} {:^10} {:^7} {:^10}'
    txt = print_format.format('Split', 'MSE', 'RMSE', 'MAPE', 'MAE')
    print_and_write(f, txt)
    for key, dataloader in zip(titles, dataloader_list):
        timstamps, targets, preds, mse, mape, l1 = run_model(model, dataloader, factors)
        all_timestamps += timstamps
        all_targets += list(targets)
        txt = print_format.format(key, round(mse, 3), round(np.sqrt(mse), 3), round(mape, 5), round(l1, 3))
        print_and_write(f, txt)
        ax.plot(
            timstamps,
            preds,
            color=split_colors[key],
            linestyle=line_styles[key],
            linewidth=2.6,
            label=key,
            alpha=0.9,
        )

        # Save test data for separate plot
        if key == 'Test':
            test_timestamps = timstamps
            test_targets = targets
            test_preds = preds

    ax.plot(
        all_timestamps,
        all_targets,
        color=split_colors['Target'],
        linestyle=line_styles['Target'],
        zorder=0,
        linewidth=2.4,
        label='Target',
        alpha=0.8,
    )
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax.set_ylabel('Price ($)', fontsize=16)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_xlim([all_timestamps[0], all_timestamps[-1]])
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}K'.format(x/1000)))
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # Plot Test only vs Target
    test_plot_path = plot_path.replace('pred.png', 'pred_test.png')
    fig2, ax2 = plt.subplots(figsize=(24, 12))
    ax2.plot(
        test_timestamps,
        test_preds,
        color=split_colors['Test'],
        linestyle=line_styles['Test'],
        linewidth=2.6,
        label='Test Prediction',
        alpha=0.9,
    )
    ax2.plot(
        test_timestamps,
        test_targets,
        color=split_colors['Target'],
        linestyle=line_styles['Target'],
        linewidth=2.4,
        label='Target',
        alpha=0.8,
    )
    ax2.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax2.set_ylabel('Price ($)', fontsize=16)
    ax2.set_xlabel('Date', fontsize=16)
    ax2.set_xlim([test_timestamps[0], test_timestamps[-1]])
    ax2.set_title('Test Set: Prediction vs Target', fontsize=18)

    # Set x-axis to show days
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Major tick every week
    ax2.xaxis.set_minor_locator(mdates.DayLocator())  # Minor tick every day
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Format: month-day
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=12)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}K'.format(x/1000)))
    ax2.grid(True, linestyle='--', alpha=0.7, which='major')
    ax2.grid(True, linestyle=':', alpha=0.4, which='minor')
    plt.tight_layout()
    plt.savefig(test_plot_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'Test-only plot saved to: {test_plot_path}')

    f.close()
