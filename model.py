import torch
import torch.nn as nn

DIM = 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(128, 4 * 4 * 4 * DIM),
                               nn.BatchNorm1d(4 * 4 * 4 * DIM), nn.ReLU(True))

        self.conv_blocks = nn.Sequential(
            
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),#8
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(inplace=True),
            #pdb.set_trace(),
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),#16
            nn.BatchNorm2d(DIM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(DIM, 3, 2, stride=2),#32
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(-1, 4 * DIM, 4, 4)
        
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        # The height and width of downsampled image
        #ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(4 * 4 * 4 * DIM, 1))

    def forward(self, img):
        feature = self.model(img)
        #print(out.shape)
        out = feature.view(feature.shape[0], -1)
        validity = self.adv_layer(out)

        return feature, validity