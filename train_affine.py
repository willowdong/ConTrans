import torch
import numpy as np
from util.utils import model_name2path
from transformers import AutoConfig
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AffineModel(torch.nn.Module):
    def __init__(self, source_dim, target_dim, use_activation=False):
        super(AffineModel, self).__init__()
        self.linear = torch.nn.Linear(source_dim, target_dim)
        self.use_activation = use_activation
        if use_activation:
            self.tanh = torch.nn.ReLU()
    def forward(self, x):
        x = self.linear(x)
        if self.use_activation:
            x = self.tanh(x)
        return x

def parser():
    import argparse
    parser = argparse.ArgumentParser(description='Train affine model')
    parser.add_argument('--source_model', type=str, default='llama-7b', help='source model')
    parser.add_argument('--target_model', type=str, default='amber-7b', help='target model')
    parser.add_argument('--source_data', type=str, help='path to source data')
    parser.add_argument('--target_data', type=str, help='path to target data')
    parser.add_argument('--save_path', type=str, help='path to save model')
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

    save_path = os.path.join(args.save_path, '{}_{}'.format(args.source_model, args.target_model))
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


    loss_record = []
    for source_layer in range(source_layers):
        for target_layer in range(target_layers):

            logger.info('Train Source layer: {} -> Target layer: {}'.format(source_layer, target_layer))

            # load data to device
            source_data = origin_source_data[:,source_layer, :]
            target_data = origin_target_data[:,target_layer, :]
            source_data = torch.from_numpy(source_data).float()
            target_data = torch.from_numpy(target_data).float()
            source_data = source_data.to(args.device)
            target_data = target_data.to(args.device)

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

            model = AffineModel(source_data.shape[1], target_data.shape[1], args.use_activation).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = torch.nn.MSELoss()
            
            # train
            best_loss = 1e10
            best_model_state_dict = None
            best_epoch = 0
            for epoch in range(args.epoch):
                for i in range(0, train_source.shape[0], args.batch_size):
                    source_batch = train_source[i:i+args.batch_size]
                    target_batch = train_target[i:i+args.batch_size]
                    pred = model(source_batch)
                    loss = criterion(pred, target_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # dev
                pred = model(dev_source)
                dev_loss = criterion(pred, dev_target)
                if dev_loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    best_model_state_dict = model.state_dict()

            # test
            pred = model(test_source)
            test_loss = criterion(pred, test_target)
            print('Best Dev Loss: {} in epoch {}'.format(best_loss.item(), best_epoch))
            print('Test Loss: {}'.format(test_loss.item()))

            # save model
            file_name = os.path.join(save_path, 'affine_{}_to_{}.pth'.format(source_layer, target_layer))
            torch.save(best_model_state_dict, file_name)

            record = {'final_train_loss':loss.item(),'best_dev_loss': best_loss.item(), 'test_loss': test_loss.item()}
            loss_record.append(record)

    # save loss record
    np.save(os.path.join(save_path, 'loss_record.npy'), loss_record)
    

if __name__=='__main__':
    args = parser()
    train_affine(args)
    print('Done!')