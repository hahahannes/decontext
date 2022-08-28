from collections import defaultdict
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils
import torch
torch.set_num_threads(20)
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time 
import random
import numpy as np
import pandas as pd 
from model import SiameseNetwork
from dataset import ThingsDataSet, WordpairDataSet, FileDataSet
from early_stopping import EarlyStopping
from loss import ContrastiveLoss, MSELoss, MAELoss
import matplotlib.pyplot as plt
from os.path import join as pjoin
import os 
os.environ['DATA_DIR'] ='/home/hannes/data/laptop_sync/Uni/master/masterarbeit/decon/DecontextEmbeddings/data'
import wandb

import sys 
sys.path.append('../..')
from helpers.plot import set_style_and_font_size
from helpers.things_evaluation.evaluate import evaluate

set_style_and_font_size()


NAME = 'mae_v10'
LOSS = 'mae'

train_dataset = None
val_dataset = None 

def save_checkpoint(state, dir, filename='checkpoint.pth.tar'):
    torch.save(state, pjoin(dir, filename))

    
def train(train_loader, model, criterion, optimizer, epoch, train_dataset):
    batch_losses = []

    for i, (embedding1, embedding2, label) in enumerate(train_loader):
        embedding1 = embedding1 #.cuda(non_blocking=True)
        embedding2 = embedding2 #.cuda(non_blocking=True)
        label = label #.cuda(non_blocking=True)

        # compute output
        output1, output2 = model(embedding1, embedding2)

        loss, loss_similar, loss_dissimilar, cosine_distance, cosine_sim = criterion(output1, output2, label)
        batch_losses.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train_batch_loss": loss, 'train_batch': epoch * len(train_loader) + i})

        print(f'batch {i} of {len(train_loader)} done')

        if False:#i%100 == 0:
            print('dog - chipmunck')
            dog_emb = train_dataset.embeddings['dog'][0]
            cat_emb = train_dataset.embeddings['chipmunk'][0]
            sim_train = train_dataset.things.loc['dog', 'chipmunk']
            output1, output2 = model(dog_emb.reshape((1,768)), cat_emb.reshape((1,768)))
            loss_contrastive, loss_similar, loss_dissimilar, cosine_distance, cosine_sim = criterion(output1, output2, torch.as_tensor([sim_train]), debug=True)
            cosine_sim = F.cosine_similarity(output1, output2).detach().numpy()[0]
            wandb.log({"dog_chipmunk_cosine_sim": cosine_sim})
            wandb.log({"dog_chipmunk_dis_loss": loss_dissimilar})
            wandb.log({"dog_chipmunk_sim_loss": loss_similar})

            print('dog - toaster')
            dog_emb = train_dataset.embeddings['toaster'][0]
            cat_emb = train_dataset.embeddings['dog'][0]
            val_train = train_dataset.things.loc['toaster', 'dog']
            output1, output2 = model(dog_emb.reshape((1,768)), cat_emb.reshape((1,768)))
            loss_contrastive, loss_similar, loss_dissimilar, cosine_distance, cosine_sim = criterion(output1, output2, torch.as_tensor([val_train]), debug=True)
            cosine_sim = F.cosine_similarity(output1, output2).detach().numpy()[0]
            wandb.log({"dog_toaster_cosine_sim": cosine_sim})
            wandb.log({"dog_toaster_dis_loss": loss_dissimilar})
            wandb.log({"dog_toaster_sim_loss": loss_similar})
    
    return np.asarray(batch_losses).mean()

def validate(val_loader, model, criterion, epoch):
    batch_losses = []
    for i, (embedding1, embedding2, label) in enumerate(val_loader):
        embedding1 = embedding1 #.cuda(non_blocking=True)
        embedding2 = embedding2 #.cuda(non_blocking=True)
        label = label #.cuda(non_blocking=True)

        output1, output2 = model(embedding1, embedding2)

        loss, loss_similar, loss_dissimilar, cosine_distance, cosine_sim = criterion(output1, output2, label)
        batch_losses.append(loss.item())
        wandb.log({"val_batch_loss": loss, 'val_batch': epoch * len(val_loader) + i})

    return np.asarray(batch_losses).mean()


def inference(model):
    model.eval()
    base_path = f'../../../data_fine_tune/things/wikidumps/decontext/bert-base/12/word/mean/all/'
    embedding_path = pjoin(base_path, 'decontext.txt')

    output_path = os.path.join(base_path, NAME + '.txt')
    with open(output_path, 'w') as output_file:
        with open(embedding_path, 'r') as embeddings_file:
            for line in embeddings_file:
                line = line.strip()
                split = line.split(';')
                word = split[0]
                n_contexts = split[1]
                embedding = split[2].split(' ')
                embedding = torch.as_tensor([float(value) for value in embedding])
                transformed_embedding1, _ = model(embedding, embedding)
                transformed_embedding1 = transformed_embedding1.detach().cpu().numpy()
    
                embedding_str = ' '.join(str(value) for value in transformed_embedding1)
                output_file.write(';'.join([word, n_contexts, embedding_str]))
                output_file.write('\n')

    val_words = list(pd.read_csv('./val_words.txt', header=None, names=['words'])['words'])
    pearson, spearman, sim_df_matrix, sim_vector = evaluate(output_path, matching='word', matching_words=val_words)
    os.remove(output_path)
    return spearman.correlation

def run(config, train_dataset, val_dataset):
    working_dir = pjoin('./output', NAME)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    embedding_dimensionality = 768
    device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    model = SiameseNetwork(embedding_dimensionality, dropout=config['dropout'], number_neurons=config['number_neurons'], number_layers=config['number_layers']).to(device)

    if LOSS == 'contrastive_loss':
        criterion = ContrastiveLoss()
    elif LOSS == 'mse':
        criterion = MSELoss()
    elif LOSS == 'mae':
        criterion = MAELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    epochs = 20
    batch_size = config['batch_size']

    train_loader =  DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=batch_size)

    val_loader =  DataLoader(val_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=batch_size)

    early_stopping = EarlyStopping(tolerance=3)

    for epoch in range(0, epochs):
        train_epoch_loss = train(train_loader, model, criterion, optimizer, epoch, train_dataset)
        val_epoch_loss = validate(val_loader, model, criterion, epoch)
        val_correlation = inference(model)

        wandb.log({'train_epoch_loss': train_epoch_loss, 'val_epoch_loss': val_epoch_loss, 'epoch': epoch, 'val_correlation': val_correlation})

        print(f'epoch {epoch} done')
        save_checkpoint({
                'next_epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
        }, working_dir)

        early_stopping(train_epoch_loss, val_epoch_loss)
        if early_stopping.early_stop:
            break

def setup_wandb():
    sweep_config = {
        "name" : NAME,
        "method" : "bayes",
        'metric': {
            'name': 'val_epoch_loss',
            'goal': 'minimize'
        },
        "parameters" : {
            "learning_rate" :{
                "min": 0.00001,
                "max": 0.1,
                "distribution": 'log_uniform_values'
            },
            "dropout": {
                'min': 0,
                'max': 0.9,
                'distribution': 'uniform'
            },
            "batch_size": {
                'values': [1000, 5000, 10000]
            },
            "weight_decay": {
                'values': [0, 1e-3, 1e-4, 1e-5, 1e-6]
            },
            'number_neurons': {
                'values': [10,50,100,300, 600]
            },
            'number_layers': {
                'values': [1,2,3]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config)
    return sweep_id

def run_wandb():
    with wandb.init(config={'loss': LOSS}, settings=wandb.Settings(start_method="fork")) as _:
        wandb.define_metric('epoch')
        wandb.define_metric('train_epoch_loss', step_metric='epoch')
        wandb.define_metric('val_epoch_loss', step_metric='epoch')

        wandb.define_metric('train_batch')
        wandb.define_metric('train_batch_loss', step_metric='train_batch')

        wandb.define_metric('val_batch')
        wandb.define_metric('val_batch_loss', step_metric='train_batch')

        config = wandb.config
        run(config, train_dataset, val_dataset)

if __name__ == '__main__':
    tuning = True 

    print('load datasets')
    train_dataset = FileDataSet(pjoin(os.environ.get('DATA_DIR'), 'retraining', 'train_fine_tune_dataset.csv'))
    val_dataset = FileDataSet(pjoin(os.environ.get('DATA_DIR'), 'retraining', 'val_fine_tune_dataset.csv'))
    print('loaded datasets')

    if tuning:
        sweep_id = setup_wandb()
        count = 100 # number of runs to execute
        wandb.agent(sweep_id, function=run_wandb, count=count)
    else:
        config = {
            'dropout': 0,
            'weight_decay': 1,
            'learning_rate': 0.1,
            'batch_size': 1000,
            'number_neurons': 50,
            'number_layers': 1
        }
        wandb.init(project=NAME)
        run(config, train_dataset, val_dataset)
    
