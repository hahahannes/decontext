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
#from torch.utils.tensorboard import SummaryWriter
import pandas as pd 
from model import SiameseNetwork
from DecontextEmbeddings.scripts.retraining.dataset import ThingsDataSet, WordpairDataSet, FileDataSet
from DecontextEmbeddings.scripts.retraining.loss import ContrastiveLoss, MSELoss
import matplotlib.pyplot as plt
from os.path import join as pjoin
import os 
import wandb

import sys 
sys.path.append('../..')
from helpers.plot import set_style_and_font_size
set_style_and_font_size()

#writer = SummaryWriter('runs/')

def save_checkpoint(state, dir, filename='checkpoint.pth.tar'):
    torch.save(state, pjoin(dir, filename))


# Extract one batch
#example_batch = next(iter(vis_dataloader))


def train(train_loader, model, criterion, optimizer, epoch, train_dataset):
    print(f'start epoch {epoch}')
    batches = []
    epoch_losses = []
    train_cosine_sims = []
    val_cosine_sims = []

    for i, (embedding1, embedding2, label) in enumerate(train_loader):
        embedding1 = embedding1 #.cuda(non_blocking=True)
        embedding2 = embedding2 #.cuda(non_blocking=True)
        label = label #.cuda(non_blocking=True)

        # compute output
        output1, output2 = model(embedding1, embedding2)

        loss, loss_similar, loss_dissimilar, cosine_distance, cosine_sim = criterion(output1, output2, label)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": loss})

        epoch_losses.append(loss.detach().numpy())
        batches.append(epoch * len(train_loader) + i)

        print(f'batch {i} of {len(train_loader)} done')

        if i%100 == 0:
            print('dog - chipmunck')
            dog_emb = train_dataset.embeddings['dog'][0]
            cat_emb = train_dataset.embeddings['chipmunk'][0]
            sim_train = train_dataset.things.loc['dog', 'chipmunk']
            output1, output2 = model(dog_emb.reshape((1,768)), cat_emb.reshape((1,768)))
            loss_contrastive, loss_similar, loss_dissimilar, cosine_distance, cosine_sim = criterion(output1, output2, torch.as_tensor([sim_train]), debug=True)
            cosine_sim = F.cosine_similarity(output1, output2).detach().numpy()[0]
            train_cosine_sims.append(cosine_sim)

            print('dog - toaster')
            dog_emb = train_dataset.embeddings['toaster'][0]
            cat_emb = train_dataset.embeddings['dog'][0]
            val_train = train_dataset.things.loc['toaster', 'dog']
            output1, output2 = model(dog_emb.reshape((1,768)), cat_emb.reshape((1,768)))
            loss_contrastive, loss_similar, loss_dissimilar, cosine_distance, cosine_sim = criterion(output1, output2, torch.as_tensor([val_train]), debug=True)
            cosine_sim = F.cosine_similarity(output1, output2).detach().numpy()[0]
            val_cosine_sims.append(cosine_sim)

    return batches, epoch_losses, train_cosine_sims, sim_train, val_cosine_sims, val_train


def validate(val_loader, model, criterion, epoch):
    epoch_losses = []
    batches = []
    for i, (embedding1, embedding2, label) in enumerate(val_loader):
        embedding1 = embedding1 #.cuda(non_blocking=True)
        embedding2 = embedding2 #.cuda(non_blocking=True)
        label = label #.cuda(non_blocking=True)

        # compute output
        output1, output2 = model(embedding1, embedding2)

        loss, loss_similar, loss_dissimilar, cosine_distance, cosine_sim = criterion(output1, output2, label)
        #writer.add_scalar('validation loss', loss, epoch * len(val_loader) + i)
        wandb.log({"val_loss": loss})
        
        epoch_losses.append(loss.detach().numpy())
        batches.append(epoch * len(val_loader) + i)
    return batches, epoch_losses

def plot_metrics(working_dir, all_batches, all_epoch_losses, val_all_batches, val_all_epoch_losses, train_epoch_mean_loss, val_epoch_mean_loss):
    # Batch losses
    fig, axes = plt.subplots(1,3, figsize=(11,3))
    axes[0].plot(all_batches, all_epoch_losses)
    axes[0].set_xlabel('Batch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train')
    axes[1].plot(val_all_batches, val_all_epoch_losses)
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation')

    # Epoch losses
    epochs = range(len(train_epoch_mean_loss))
    axes[2].plot(epochs, train_epoch_mean_loss, label='Train')
    axes[2].plot(epochs, val_epoch_mean_loss, label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_xticks(epochs)
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(pjoin(working_dir, 'loss.pdf'), bbox_inches='tight')

def plot_cosine(working_dir, cosines_similar, cosines_dissimilar, true_cosine_similar, true_cosine_dissimilar):
    fig, axes = plt.subplots(1,2, figsize=(11,3))

    axes[0].plot(range(len(cosines_similar)), cosines_similar, label='cosine sim')
    axes[0].set_xlabel('batch')
    axes[0].set_ylabel('cosine sim')
    axes[0].set_title('Dog - Chipmunk')
    axes[0].axhline(true_cosine_similar, label='true similarity', c='green')

    axes[1].plot(range(len(cosines_dissimilar)), cosines_dissimilar, label='cosine sim')
    axes[1].set_xlabel('batch')
    axes[1].set_ylabel('cosine sim')
    axes[1].set_title('Dog - Toast')
    axes[1].axhline(true_cosine_dissimilar, label='true similarity', c='green')
    plt.savefig(pjoin(working_dir, 'cosines.png'))


def run(loss, name, config):
    working_dir = pjoin('.', name)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    embedding_dimensionality = 768
    device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    model = SiameseNetwork(embedding_dimensionality).to(device)

    if loss == 'contrastive_loss':
        criterion = ContrastiveLoss()
    elif loss == 'mse':
        criterion = MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr = 0.1, weight_decay=1e-5)
    epochs = 15
    batch_size = 1000
    embeddings_path = '../../../data_fine_tune/things/wikidumps/decontext/bert-base/12/extractions.txt'


    train_dataset = FileDataSet('train_fine_tune_dataset.csv', embeddings_path, './train_words.txt')
    #train_dataset = ThingsDataSet(embeddings_path, './train_words.txt')

    train_loader =  DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=batch_size)

    val_dataset = FileDataSet('val_fine_tune_dataset.csv', embeddings_path, './val_words.txt')
    #val_dataset = ThingsDataSet(embeddings_path, './val_words.txt')
    val_loader =  DataLoader(val_dataset,
                        shuffle=True,
                        num_workers=1,
                        batch_size=batch_size)

    all_batches = []
    all_epoch_losses = []
    val_all_batches = []
    val_all_epoch_losses = []
    all_ctrain_cosine_sims = []
    all_val_cosine_sims = []

    train_epoch_mean_loss = []
    val_epoch_mean_loss = []

    for epoch in range(0, epochs):
        batches, epoch_losses, train_cosine_sims, train_sim, val_cosine_sims, val_sim = train(train_loader, model, criterion, optimizer, epoch, train_dataset)
        all_batches += batches 
        all_epoch_losses += epoch_losses
        all_ctrain_cosine_sims += train_cosine_sims
        all_val_cosine_sims += val_cosine_sims
        train_epoch_mean_loss.append(np.asarray(epoch_losses).mean())

        val_batches, val_epoch_losses = validate(val_loader, model, criterion, epoch)
        val_all_batches += val_batches
        val_all_epoch_losses += val_epoch_losses        
        val_epoch_mean_loss.append(np.asarray(val_epoch_losses).mean())

        #plot_metrics(working_dir, all_batches, all_epoch_losses, val_all_batches, val_all_epoch_losses, train_epoch_mean_loss, val_epoch_mean_loss)
        #plot_cosine(working_dir, train_cosine_sims, val_cosine_sims, train_sim, val_sim)
        
        print(f'epoch {epoch} done')
        save_checkpoint({
                'next_epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
        }, working_dir)

def setup_wandb(loss):
    wandb.init(project="retraining")

    wandb.init({'loss': loss})

    sweep_config = {
        "name" : "sweep1",
        "method" : "random",
        "parameters" : {
            "epochs" : {
                "values" : [10, 20, 50]
            },
            "learning_rate" :{
                "min": 0.0001,
                "max": 0.1
            },
            "dropout": {
                'values': [0.05, 0.1, 0.2]
            },
            "batch_size": {
                'values': [10, 100, 1000, 10000]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config)


if __name__ == '__main__':
    name = 'mse_new'
    loss = 'mse'
    
    setup_wandb(loss)

    with wandb.init() as run:
        config = wandb.config
        run(loss, name, config)


    count = 5 # number of runs to execute
    wandb.agent(sweep_id, function=train, count=count)
    
    
