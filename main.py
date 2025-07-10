import argparse
import json
import pickle
import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from quality_metrics import *

# Ensure device is set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper function to get config
def get_config(config_path, common_config_path="configs/common.json"):
    with open(common_config_path, 'r') as f:
        config_common = json.load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)

    for key in config_common.keys():
        if key not in config:
            config[key] = config_common[key]
    
    return ParamsDynamic(config)

# PDE functions
def gs_speed_norm(v, v_t, v_d):
    """Residual of the PDE, v is speed"""
    return v_d - 2 * v * v_d - v_t

# Data Loader Class from Notebook
class NGSIMCustomDataLoader(Dataset):
    def __init__(self, config, df, batch_size=32, mode='train', shuffle=False):
        self.df = df
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = torch.from_numpy(df[['time','distance']].values).float().to(device)
        self.x_raw = torch.from_numpy(df[['time_raw','distance_raw']].values).float().to(device)
        
        self.y = torch.from_numpy(df[config.mode].values).float().to(device)
        self.y_raw = torch.from_numpy(df[[f'{c}_raw' for c in config.mode]].values).float().to(device)
        
        if self.config.train_sample_method == "random":
            np.random.seed(config.random_seed)
            all_idxs = np.arange(len(df))
            np.random.shuffle(all_idxs)
            split_n = int(config.train_sample_p * len(df))
            self.train_idxs = all_idxs[:split_n]
            print(f'Random Data: {int(config.train_sample_p * 100)}% ({split_n} of {len(df)}) samples were assigned for training.')
        
        if mode == 'train':
            self.idxs = self.train_idxs
        
        self.create_batch()

    def create_batch(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.data = []
        for index in np.array_split(self.idxs, len(self.idxs)//self.batch_size+1):
            x = self.x[index]
            x_raw = self.x_raw[index]
            y = self.y[index]
            y_raw = self.y_raw[index]
            self.data.append([x, x_raw, y, y_raw])

# Model Definitions from Notebook
class NN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hs = self.config.nn_hs
        n_layers = self.config.nn_n_layers
        self.bn_hidden = nn.BatchNorm1d(hs)
        self.fc_in = nn.Linear(2, hs)
        self.fc_mids = nn.ModuleList([nn.Linear(hs, hs) for i in range(n_layers)])
        self.fc_out = nn.Linear(hs, len(config.mode))

        self.init_weights()
        self.ub = torch.Tensor([[float(self.config.time['max']), float(self.config.distance['max'])]]).to(device)
        self.lb = torch.Tensor([[0.,  0.]]).to(device)
        self.isRaw = True

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.01)

    def forward(self, x):
        if self.isRaw:
            x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        x = self.fc_in(x)
        for i, l in enumerate(self.fc_mids):
            if self.config.nn_enable_residual:
                x = torch.tanh(l(x)) + x
            else:
                x = torch.tanh(l(x))
        x = self.fc_out(x)
        return F.relu(x)

def get_pde_loss(model, config):
    time_random = torch.from_numpy(np.random.uniform(config.time['min'], config.time['max'], (config.n_random_inputs, 1))).float().to(device)
    distance_random = torch.from_numpy(np.random.uniform(config.distance['min'], config.distance['max'], (config.n_random_inputs, 1))).float().to(device)
    
    time_random.requires_grad = True
    distance_random.requires_grad = True
    x_random = torch.cat([time_random, distance_random], dim=-1)
    
    y_rand_pred = model(x_random)
    
    u = y_rand_pred
    u_t = torch.autograd.grad(u, time_random, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_d = torch.autograd.grad(u, distance_random, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    
    loss_pde = {}
    for pde_model_name in config.physical_model:
        pde_res = globals()[pde_model_name](u, u_t, u_d)
        loss_pde[pde_model_name] = (pde_res ** 2).mean()
    return loss_pde

def train_model(args):
    config_path = args.config
    config_name = os.path.basename(config_path).replace('.json', '')
    params = get_config(config_path)

    # Override n_epochs if provided in args
    if args.n_epochs:
        params.dict['n_epochs'] = args.n_epochs

    print(f"Training with config: {config_name}")
    print(f"Epochs: {params.n_epochs}")
    
    # Load data
    df_all = pd.read_csv(f'data/pinn_data_{params.dataset_name}_norm.csv.gz')
    params.dict['time'] = {'max': df_all.time_raw.max(), 'min':0}
    params.dict['distance'] = {'max': df_all.distance_raw.max(), 'min':0}
    params.dict['speed'] = {'max': df_all.speed_raw.max(), 'min':0}
    
    dataloader = NGSIMCustomDataLoader(params, df_all, batch_size=params.batch_size, shuffle=True)
    
    # Model
    model = NN(params).to(device)
    
    # Optimizer
    optimizer = Lamb(model.parameters(), lr=params.lr, weight_decay=params.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=params.scheduler_patience, factor=0.5, min_lr=1e-8)
    loss_fn_mse = nn.MSELoss()

    # Training Loop
    log_dir = f'logs/{config_name}/version_{get_next_version(f"logs/{config_name}")}'
    os.makedirs(log_dir, exist_ok=True)
    
    best_loss = float('inf')
    best_model_state = None
    
    print(f"Logging to {log_dir}")
    model.train()
    loop = tqdm.tqdm(range(1, params.n_epochs + 1), total=params.n_epochs)
    
    for epoch in loop:
        epoch_summary = {'loss_nn':[], 'loss':[]}
        for x, x_raw, y, y_raw in dataloader.data:
            loss = 0
            postfix_dict = {}

            if params.loss_nn:
                y_pred = model(x_raw)
                loss_nn = loss_fn_mse(y_raw, y_pred)
                loss += loss_nn
                epoch_summary['loss_nn'].append(loss_nn.item())
                postfix_dict['nn'] = loss_nn.item()

            if params.loss_pde:
                pde_losses = get_pde_loss(model, params)
                for key, value in pde_losses.items():
                    loss += value
                    if f'loss_{key}' not in epoch_summary:
                        epoch_summary[f'loss_{key}'] = []
                    epoch_summary[f'loss_{key}'].append(value.item())
                    postfix_dict[key] = value.item()
            
            epoch_summary['loss'].append(loss.item())
            postfix_dict['loss'] = loss.item()
            loop.set_postfix(postfix_dict)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = np.mean(epoch_summary['loss'])
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(log_dir, 'checkpoint_best.pt'))

    print("Training finished.")

    # Evaluation
    print("Evaluating model...")
    model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint_best.pt')))
    model.eval()
    
    df_x = torch.from_numpy(df_all[['time_raw', 'distance_raw']].values).float().to(device)
    with torch.no_grad():
        df_y_pred = model(df_x)

    for i, mode in enumerate(params.mode):
        df_all[f'{mode}_pred'] = df_y_pred[:, i].detach().cpu().numpy()
    
    df_all.to_csv(os.path.join(log_dir, 'predictions.csv.gz'), index=False, compression='gzip')

    # Calculate and save metrics
    results = {}
    for i, mode in enumerate(params.mode):
        img_real = pd.pivot_table(df_all, values=f'{mode}_raw', index='distance_raw', columns='time_raw').values
        img_pred = pd.pivot_table(df_all, values=f'{mode}_pred', index='distance_raw', columns='time_raw').values
        
        img_real = np.expand_dims(img_real, axis=-1)
        img_pred = np.expand_dims(img_pred, axis=-1)

        results[mode] = {
            'mse': mse(img_real, img_pred),
            'mape': mape(img_real, img_pred),
            'psnr': psnr(img_real, img_pred),
            'fsim': fsim(img_real, img_pred)
        }
    
    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
        
    print("Metrics saved.")
    
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4))
    sns.heatmap(pd.pivot_table(df_all, values=f'{params.mode[0]}_pred', index='distance_raw', columns='time_raw'), cmap="jet_r", ax=ax, vmin=0, vmax=80) 
    ax.set_title(f"{config_name} | MSE: {results[params.mode[0]]['mse']:.2f}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.invert_yaxis()
    plt.savefig(os.path.join(log_dir, 'prediction.png'), dpi=300)
    plt.show()

    print(f"Evaluation finished. Results saved in {log_dir}")
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN model for traffic flow.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--n_epochs", type=int, help="Number of epochs to run, overrides config file.")
    args = parser.parse_args()

    # Create common config if it doesn't exist
    if not os.path.exists("configs/common.json"):
        config_common = {
            'mode': ['speed'],
            'physical_model': ['gs_speed_norm'],
            'loss_nn': True,
            'loss_pde': False,
            'train_sample_p': 0.05,
            'main_path': './',
            'random_seed': 42,
            'nn_enable_residual': False,
            'nn_hs': 32,
            'nn_n_layers': 8,
            'n_random_inputs': 4000,
            'n_epochs': 3000,
            'scheduler_patience': 200,
            'batch_size': 4096,
            'optimizer': 'lamb',
            'lr': 4e-3,
            'wd': 2e-4,
        }
        with open('configs/common.json', 'w', encoding='utf-8') as f:
            json.dump(config_common, f, indent=4)
    
    train_model(args)
