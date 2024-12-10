import os
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from statistics import mean

# Importing specific masking functions and other utilities
from src.utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from src.utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams

# Importing the DiffPatchImputer class
from src.imputers.DiffPatchImputer import DiffPatchImputer

def load_model(use_model, model_config):
    if use_model == 0:
        return DiffPatchImputer(**model_config).cuda()
    elif use_model == 1:
        return DiffPatchImputer(**model_config).cuda()
    elif use_model == 2:
        return DiffPatchImputer(**model_config).cuda()
    else:
        raise ValueError('Invalid model selection')

def main(config_path, ckpt_iter, num_samples):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract configurations
    gen_config = config['gen_config']
    train_config = config['train_config']
    trainset_config = config['trainset_config']
    diffusion_config = config['diffusion_config']
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    # Determine model configuration based on the use_model flag
    model_config_key = 'DiffPatch_config'
    model_config = config[model_config_key]

    # Initialize model
    model = load_model(train_config['use_model'], model_config)
    print_size(model)

    # Setup output directory
    output_dir = os.path.join(gen_config['output_directory'], 'T{}_BET0{}_BETT{}'.format(diffusion_config["T"], diffusion_config["beta_0"], diffusion_config["beta_T"]))
    os.makedirs(output_dir, exist_ok=True)
    os.chmod(output_dir, 0o775)

    # Load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(gen_config['ckpt_path'])
    checkpoint_path = os.path.join(gen_config['ckpt_path'], f'{ckpt_iter}.pkl')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    data = np.load(trainset_config['test_data_path'])
    data = np.split(data, 5, 0)
    data = np.array(data)
    data = torch.from_numpy(data).float().cuda()

    all_mse = []
    for i, batch in enumerate(data):
        mask_func = get_mask_mnr if train_config['masking'] == 'mnr' else get_mask_bm if train_config['masking'] == 'bm' else get_mask_rm
        mask = mask_func(batch[0], train_config['missing_k'])
        mask = mask.transpose(1, 0).repeat(batch.size(0), 1, 1).type(torch.float).cuda()
        batch = batch.transpose(1, 2)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        generated_audio = sampling(model, (num_samples, batch.size(1), batch.size(2)), diffusion_hyperparams, cond=batch, mask=mask, only_generate_missing=train_config['only_generate_missing'])

        end_event.record()
        torch.cuda.synchronize()

        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        outfile = f'imputation{i}.npy'
        np.save(os.path.join(output_dir, outfile), generated_audio)
        outfile = f'original{i}.npy'
        np.save(os.path.join(output_dir, outfile), batch)
        outfile = f'mask{i}.npy'
        np.save(os.path.join(output_dir, outfile), mask)

        mse = mean_squared_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
        all_mse.append(mse)

    print(f'Total MSE: {mean(all_mse)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json', help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max', help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=500, help='Number of utterances to be generated')
    args = parser.parse_args()

    main(args.config, args.ckpt_iter, args.num_samples)