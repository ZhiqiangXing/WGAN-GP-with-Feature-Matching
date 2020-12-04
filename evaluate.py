import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torch.autograd as autograd

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import pdb
from model import Generator

PATH = "records-wgan-gp/"
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
def get_sample_list(G, use_cuda=False):
    all_samples = []
    for i in range(10):
        samples_100 = torch.randn(100, 128)
        if use_cuda:
            samples_100 = samples_100.cuda()
            G.cuda()
        samples_100 = autograd.Variable(samples_100, volatile=True)
        pdb.set_trace()
        print(G(samples_100).cpu().data.numpy()[0])
        all_samples.append(G(samples_100).cpu().data.numpy())
    all_samples = np.concatenate(all_samples, axis=0)
    return list(all_samples)
class EvaluateDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    generator = Generator()
    generator.load_state_dict(torch.load(PATH + 'generator.pth'))
    generator.eval()
    sample_list= get_sample_list(generator)
    evaluate_dataset = EvaluateDataset(sample_list)

    print ("Calculating Inception Score...")
    print (inception_score(evaluate_dataset, cuda=True, batch_size=32, resize=True, splits=10))