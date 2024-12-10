import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

# Importing specific masking functions and other utilities
from src.utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from src.utils.util import get_mask_mnr, get_mask_bm, get_mask_rm

# Importing the DiffPatchImputer class
from src.imputers.DiffPatchImputer import DiffPatchImputer

def train_model(output_directory, ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging, learning_rate, use_model, only_generate_missing, masking, missing_k):
    # Set up local path and output directory
    local_path = f"T{diffusion_config['T']}_beta0{diffusion_config['beta_0']}_betaT{diffusion_config['beta_T']}"
    output_directory = os.path.join(output_directory, local_path)
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # Map diffusion hyperparameters to GPU
    diffusion_hyperparams = {k: v.cuda() for k, v in calc_diffusion_hyperparams(**diffusion_config).items() if k != "T"}

    # Predefine model
    model = DiffPatchImputer(**model_config).cuda()
    print_size(model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if exists
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        model_path = os.path.join(output_directory, f'{ckpt_iter}.pkl')
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Successfully loaded model at iteration {ckpt_iter}')
    else:
        print('No valid checkpoint model found, start training from initialization.')

    # Load and prepare data
    training_data = np.load(trainset_config['train_data_path'])
    training_data = torch.from_numpy(training_data).float().cuda()
    print('Data loaded')

    # Training loop
    for n_iter in range(ckpt_iter + 1, n_iters + 1):
        for batch in training_data:
            mask_func = get_mask_rm if masking == 'rm' else get_mask_mnr if masking == 'mnr' else get_mask_bm
            mask = mask_func(batch[0], missing_k)
            mask = mask.permute(1, 0).repeat(batch.size(0), 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1)

            # Back-propagation
            optimizer.zero_grad()
            loss = training_loss(model, nn.MSELoss(), (batch, batch, mask, loss_mask), diffusion_hyperparams, only_generate_missing=only_generate_missing)
            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print(f"iteration: {n_iter} \tloss: {loss.item()}")

            # Save checkpoint
            if n_iter % iters_per_ckpt == 0:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, f'{n_iter}.pkl'))
                print(f'model at iteration {n_iter} is saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/DiffPatch.json', help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_config = config["train_config"]
    trainset_config = config["trainset_config"]
    diffusion_config = config["diffusion_config"]
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    model_config_key = 'DiffPatch_config'  # Update the key to match the new config name
    model_config = config[model_config_key]

    train_model(**train_config)