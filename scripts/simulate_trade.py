import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import numpy as np
import seaborn as sns
from utils import io_tools
from utils.trade import trade
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_theme(style='whitegrid', context='paper', font_scale=2)
palette = sns.color_palette('muted')

ROOT = io_tools.get_root(__file__, num_returns=2)

LABEL_DICT = {
    'cmamba': 'CryptoMamba',
    'lstm': 'LSTM',
    'lstm_bi': 'Bi-LSTM',
    'gru': 'GRU',
    'smamba': 'S-Mamba',
    'itransformer': 'iTransformer',
    'linear_crypto': 'LinearCrypto',
    'linear_crypto_nv': 'LinearCrypto',
}

def get_args():
    parser = ArgumentParser()
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
        required=True,
        help="Config name(s). Use comma to separate multiple configs (e.g., 'linear_crypto,cmamba_nv').",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        help="Checkpoint path. For single config only. For multiple configs, use --ckpt_paths or auto-find latest.",
    )
    parser.add_argument(
        "--ckpt_paths",
        default=None,
        type=str,
        help="Comma-separated checkpoint paths for multiple configs (e.g., 'path1.ckpt,path2.ckpt').",
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

    parser.add_argument(
        "--balance",
        type=float,
        default=100,
        help="initial money",
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=2,
    )

    parser.add_argument(
        "--split",
        type=str,
        default='test',
        choices={'test', 'val', 'train'},
    )

    parser.add_argument(
        "--trade_mode",
        type=str,
        default='smart',
        choices={'smart', 'smart_w_short', 'vanilla', 'no_strategy'},
    )

    args = parser.parse_args()
    return args

def find_latest_checkpoint(config_name):
    """Find the latest checkpoint for a given config name."""
    import glob
    # Try to find in logs directory
    log_pattern = f'{ROOT}/logs/*/version_*/checkpoints/*.ckpt'
    checkpoints = glob.glob(log_pattern)

    # Filter by config name (match the log directory name)
    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{config_name}.yaml')
    exp_name = config.get('name', config_name)

    matching_ckpts = [ckpt for ckpt in checkpoints if exp_name in ckpt]

    if not matching_ckpts:
        raise FileNotFoundError(f"No checkpoint found for config '{config_name}'. Expected in logs/{exp_name}/version_*/checkpoints/")

    # Return the most recent checkpoint
    latest_ckpt = max(matching_ckpts, key=os.path.getmtime)
    print(f"Auto-found checkpoint for {config_name}: {latest_ckpt}")
    return latest_ckpt

def load_model(config, ckpt_path, config_name=None):
    if ckpt_path is None:
        if config_name:
            ckpt_path = find_latest_checkpoint(config_name)
        else:
            ckpt_path = f'{ROOT}/checkpoints/{config_name}.ckpt'
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


def init_dirs(args, name):
    # Extract version directory from checkpoint path
    # e.g., logs/CMamba_Binance/version_1/checkpoints/last.ckpt -> logs/CMamba_Binance/version_1
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
        if 'checkpoints' in ckpt_path:
            version_dir = os.path.dirname(os.path.dirname(ckpt_path))
        else:
            version_dir = os.path.dirname(ckpt_path)

        # Create results subdirectory in the version folder
        path = f'{version_dir}/results'
    else:
        # Fallback to old behavior if no checkpoint path is provided
        path = f'{ROOT}/Results/{name}/{args.config}'
        if name == 'all':
            path = f'{ROOT}/Results/all/'

    if not os.path.isdir(path):
        os.makedirs(path)
    print(f'Saving results to: {path}')

    return path

def max_drawdown(prices):
    prices = np.array(prices)
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    mdd = drawdown.min()
    return -mdd


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
            if factors is not None:
                timetamps += [float(x) for x in list(batch.get('Timestamp_orig').numpy().reshape(-1))]
            else:
                timetamps += [float(x) for x in list(ts)]

    if factors is not None:
        scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
        shift = factors.get(model.y_key).get('min')
        target_list = [x * scale + shift for x in target_list]
        preds_list = [x * scale + shift for x in preds_list]

    targets = np.asarray(target_list)
    preds = np.asarray(preds_list)

    return timetamps, targets, preds


if __name__ == '__main__':
    args = get_args()
    init_dir_flag = False
    results_path = None
    colors = ['orange', 'darkblue', 'yellowgreen', 'crimson', 'darkviolet', 'magenta']

    # Parse config names (support comma-separated list)
    if args.config == 'all':
        config_list = [x.replace('.ckpt', '') for x in os.listdir(f'{ROOT}/checkpoints/') if '_nv.ckpt' in x]
    elif args.config == 'all_v':
        config_list = [x.replace('.ckpt', '') for x in os.listdir(f'{ROOT}/checkpoints/') if '_v.ckpt' in x]
        results_path = init_dirs(args, 'all')
    else:
        # Support comma-separated config names
        config_list = [c.strip() for c in args.config.split(',')]
        if len(config_list) == 1:
            init_dir_flag = True

    # Parse checkpoint paths (support comma-separated list)
    ckpt_path_list = []
    if args.ckpt_paths:
        # Multiple checkpoint paths provided
        ckpt_path_list = [p.strip() for p in args.ckpt_paths.split(',')]
        if len(ckpt_path_list) != len(config_list):
            raise ValueError(f"Number of ckpt_paths ({len(ckpt_path_list)}) must match number of configs ({len(config_list)})")
    elif args.ckpt_path:
        # Single checkpoint path for all configs
        ckpt_path_list = [args.ckpt_path] * len(config_list)
    else:
        # Auto-find checkpoints
        ckpt_path_list = [None] * len(config_list)

    plt.figure(figsize=(15, 10))
    for idx, (conf, c) in enumerate(zip(config_list, colors)):
        config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{conf}.yaml')
        if init_dir_flag:
            init_dir_flag = False
            results_path = init_dirs(args, config.get('name', args.expname))
        data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")

        # Use the corresponding checkpoint path for this config
        ckpt_path = ckpt_path_list[idx]
        model, normalize = load_model(config, ckpt_path, config_name=conf) 

        use_volume = config.get('use_volume', False)
        test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
        data_module = CMambaDataModule(data_config,
                                        train_transform=test_transform,
                                        val_transform=test_transform,
                                        test_transform=test_transform,
                                        batch_size=args.batch_size,
                                        distributed_sampler=False,
                                        num_workers=args.num_workers,
                                        normalize=normalize,
                                        )
        
        if args.split == 'test':
            test_loader = data_module.test_dataloader()
        if args.split == 'val':
            test_loader = data_module.val_dataloader()
        if args.split == 'train':
            test_loader = data_module.train_dataloader()

        factors = None
        if normalize:
            factors = data_module.factors
        timstamps, targets, preds = run_model(model, test_loader, factors)

        data = test_loader.dataset.data
        tmp = data.get('Close')
        time_key = 'Timestamp'
        if normalize:
            time_key = 'Timestamp_orig'
            scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
            shift = factors.get(model.y_key).get('min')
            data[model.y_key] = data[model.y_key] * scale + shift

        balance, balance_in_time = trade(data, time_key, timstamps, targets, preds, 
                                         balance=args.balance, mode=args.trade_mode, 
                                         risk=args.risk, y_key=model.y_key)

        print(f'{conf} -- Final balance: {round(balance, 2)}')
        print(f'{conf} -- Maximum Draw Down : {round(max_drawdown(balance_in_time) * 100, 2)}')

        label = conf.replace("_nv", "").replace("_v", "")
        label = LABEL_DICT.get(label, label)
        tmp = [timstamps[0] - 24 * 60 * 60] + timstamps
        tmp = [datetime.fromtimestamp(int(x)) for x in tmp]
        sns.lineplot(x=tmp, 
                     y=balance_in_time, 
                     color=c, 
                     zorder=0, 
                     linewidth=2.5, 
                     label=label)

    name = config.get('name', args.expname)

    # Determine save path based on number of configs
    if len(config_list) > 1:
        # Multiple configs - save to comparison directory
        comparison_dir = f'{ROOT}/Results/comparison'
        if not os.path.isdir(comparison_dir):
            os.makedirs(comparison_dir)
        config_names = '_vs_'.join([c.replace('_nv', '').replace('_v', '') for c in config_list[:3]])  # Limit to 3 names
        if len(config_list) > 3:
            config_names += f'_and_{len(config_list)-3}_more'
        plot_path = f'{comparison_dir}/{config_names}_{args.split}_{args.trade_mode}.jpg'
    else:
        # Single config - use original logic
        if results_path is not None:
            if args.trade_mode == 'no_strategy':
                plot_path = f'{results_path}/balance_{args.split}.jpg'
            else:
                plot_path = f'{results_path}/balance_{args.split}_{args.trade_mode}.jpg'
        else:
            # Fallback to old behavior
            if args.trade_mode == 'no_strategy':
                plot_path = f'./balance_{args.split}.jpg'
            else:
                plot_path = f'{ROOT}/Results/{name}/{args.config}/balance_{args.split}_{args.trade_mode}.jpg'
    plt.xticks(rotation=30)
    plt.axhline(y=100, color='r', linestyle='--')

    if len(config_list) == 1:
        ax = plt.gca()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        plt.title(f'Balance in time (final: {round(balance, 2)})')
    else:
        plt.title(f'Net Worth in Time')

    # matplotlib.rcParams.update({'font.size': 100})
    plt.xlim([tmp[0], tmp[-1]])
    plt.ylabel('Balance ($)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Trade simulation plot saved to: {plot_path}')
