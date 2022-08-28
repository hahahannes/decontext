import torch
from torch import optim
import torch.nn.functional as F

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, neg_margin=1, pos_margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def forward(self, output1, output2, label, debug=False):
      # Calculate the euclidean distance and calculate the contrastive loss
        #euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        cosine_sim = F.cosine_similarity(output1, output2)
        cosine_distance = 1 - cosine_sim

        # CosSim = 1 -> CosDis = 1-1 = 0
        # CosSim = 0 -> CosDis = 1-0 = 1
        # CosSim = -1 -> CosDis = 1- -1 = 2
        
        # TODO cosine similarity - spose dot similarity -> desired_distance = 1 - label
        label = label.reshape((1,len(label)))[0]

        loss_similar = (label) * torch.pow(cosine_distance, 2)
        loss_dissimilar = (1-label) * torch.pow(torch.clamp(1 - cosine_distance, min=0.0), 2)
        
        if debug:
            print('cosine sim')
            print(cosine_sim)
            print('cosine dis')
            print(cosine_distance)
            print('label')
            print(label)
            print('similar')
            print(loss_similar)
            print('dissimilar')
            print(loss_dissimilar)
        
        loss_contrastive = torch.mean(loss_similar + loss_dissimilar)
        
        return loss_contrastive, loss_similar, loss_dissimilar, cosine_distance, cosine_sim

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output1, output2, label, debug=False):
        cosine_sim = F.cosine_similarity(output1, output2)
        label = label.reshape((1,len(label)))[0]
        loss = torch.mean(torch.pow(label - cosine_sim, 2))
        
        return loss, None, None, None, None

class MAELoss(torch.nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output1, output2, label, debug=False):
        cosine_sim = F.cosine_similarity(output1, output2)
        label = label.reshape((1,len(label)))[0]
        loss = torch.mean(torch.abs(label - cosine_sim))
        
        return loss, None, None, None, None

  
