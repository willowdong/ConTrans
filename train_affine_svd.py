import torch
import numpy as np
from util.utils import model_name2path
from transformers import AutoConfig
import logging
import os
from scipy.linalg import svd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def solve_via_svd(X, Y):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    
    Sigma_inv = np.diag(1 / Sigma)
    
    W = Y.T @ U @ Sigma_inv @ VT
    
    return W

def parser():
    import argparse
    parser = argparse.ArgumentParser(description='Train affine model')
    parser.add_argument('--source_model', type=str, default='llama-7b', help='source model')
    parser.add_argument('--target_model', type=str, default='amber-7b', help='target model')
    parser.add_argument('--source_data', type=str,  help='path to source data')
    parser.add_argument('--target_data', type=str,  help='path to target data')
    parser.add_argument('--save_path', type=str, default='vectors/affine_weights', help='path to save model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=47, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--use_activation', default=False, help='use activation function')
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def train_affine(args):
    
    if args.source_model == args.target_model:
        logger.info('Source model and target model are the same, skip training')
        return
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    #set seed
    set_seed(args.seed)

    assert args.source_model != args.target_model, 'source model and target model should be different'
    assert args.source_model in args.source_data and args.target_model in args.target_data, 'source model and target model should be in source data and target data'

    source_model_config = AutoConfig.from_pretrained(model_name2path[args.source_model])
    target_model_config = AutoConfig.from_pretrained(model_name2path[args.target_model])
    if 'llama' in args.source_model or 'pythia' in args.source_model or 'amber' in args.source_model:
        source_layers = source_model_config.num_hidden_layers
    else:
        raise NotImplementedError

    if 'llama' in args.target_model or 'pythia' in args.target_model or 'amber' in args.target_model:
        target_layers = target_model_config.num_hidden_layers
    else:
        raise NotImplementedError

    logger.info('Source model: {} with {} layers, Target model: {} with {} layers'.format(args.source_model,source_layers, args.target_model, target_layers))

    save_path = os.path.join(args.save_path, '{}_{}_svd'.format(args.source_model, args.target_model))
    if not args.use_activation:
        save_path = save_path+'_wiki_split'
    else:
        save_path = save_path+'_wiki_split_activation'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    origin_source_data = np.load(args.source_data)
    origin_target_data = np.load(args.target_data)
    train_ratio = 0.8
    dev_ratio = 0.1

    criterion = torch.nn.MSELoss()

    loss_record = []
    for source_layer in range(24,25):
        for target_layer in range(30,31):

            if source_layer>target_layer or target_layer>source_layer*2+3:
                continue

            logger.info('Train Source layer: {} -> Target layer: {}'.format(source_layer, target_layer))

            # load data to device
            source_data = origin_source_data[:,source_layer, :]
            target_data = origin_target_data[:,target_layer, :]

            # split data to train, dev, test
            train_size = int(source_data.shape[0] * train_ratio)
            dev_size = int(source_data.shape[0] * dev_ratio)
            test_size = source_data.shape[0] - train_size - dev_size
            train_source = source_data[:train_size]
            train_target = target_data[:train_size]
            dev_source = source_data[train_size:train_size+dev_size]
            dev_target = target_data[train_size:train_size+dev_size]
            test_source = source_data[train_size+dev_size:]
            test_target = target_data[train_size+dev_size:]

            print('Train size: {}, Dev size: {}, Test size: {}'.format(train_size, dev_size, test_size))
            
            W = solve_via_svd(train_source, train_target)
            
            # # test
            pred = test_source @ W.T
            pred = torch.from_numpy(pred).float()
            test_target = torch.from_numpy(test_target).float()
            test_loss = criterion(pred, test_target)
            print('Test Loss: {}'.format(test_loss.item()))

            # save model
            W = torch.from_numpy(W).float()
            file_name = os.path.join(save_path, 'affine_{}_to_{}.pth'.format(source_layer, target_layer))
            torch.save(W, file_name)

    np.save(os.path.join(save_path, 'loss_record.npy'), loss_record)

if __name__=='__main__':
    args = parser()
    train_affine(args)
    print('Done!')