import os, argparse
import torch
from torch import nn
from torchvision import datasets
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.datasets import mnist
from torchvision.utils import save_image

# from torchvision.models.inception import inception_v3

import torchvision.models as models
from models import model_dict
from DCGAN_2 import Generator as DCGAN_G
from CGAN import Generator as CGAN_G
from DCGAN_2_vbn import Generator as DCGAN_vbn_G
from CGAN_vbn import Generator as CGAN_vbn_G

import numpy as np
from scipy.stats import entropy


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--model", type=str, default="ResNet18", help="model name")
parser.add_argument("--generator_path", type=str, default="", help="pretrained generator")
parser.add_argument("--gan_model", type=str, default="DCGAN_2", choices=["DCGAN_2", "DCGAN_2_vbn", "CGAN", "CGAN_vbn"], help="kind of generator that we will use to generate image")
parser.add_argument("--ckpt_dir", type=str, default="./classifier_ckpt", help="checkpoint path of trained classifier")

opt = parser.parse_args()


def inception_score(path=opt.generator_path, path_classifier =opt.ckpt_dir , cuda=True, batch_size=64, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    # assert batch_size > 0
    # assert N > batch_size
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    if torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    opt.img_shape = (opt.channels, opt.img_size, opt.img_size)
    # Set up dataloader
    ## change to generated img
    ### ??? ???????????? ?????????. -> Generated img??? ???????????? ??? ?????? GAN skpt??? ???????????? random variable??? ?????? ?????? img generate
    #### Question -> ??? ?????? ???????????? ????????????? -> MNIST dataset ????????????
    if opt.gan_model == "DCGAN_2":
        generator = DCGAN_G(opt)
    elif opt.gan_model == "DCGAN_2_vbn":
        generator = DCGAN_vbn_G(opt)
    elif opt.gan_model == "CGAN":
        generator = CGAN_G(opt)
    elif opt.gan_model == "CGAN_vbn":
        generator = CGAN_vbn_G(opt)

    ckpt_G = torch.load(path) #opt ??????
    try:
        generator.load_state_dict(ckpt_G['generator'])
    except RuntimeError as e:
        print('wrong checkpoint')
    else:    
        print('generator checkpoint is loaded !')

    opt.device = torch.device("cuda")
    generator.to(opt.device)

    ### Generator??? ?????? imgs??????????????? ### 
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # print(len(dataloader))
    imgs_result = torch.empty((64,1,32,32)).to(opt.device) ##?????? ?????????!
    for i, (imgs_input, _) in enumerate(dataloader):
        z = Variable(Tensor(np.random.normal(0, 1, (imgs_input.shape[0], opt.latent_dim))))

        if opt.gan_model == "DCGAN_2" or opt.gan_model == "DCGAN_2_vbn":
            imgs = generator(z)
        else:
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
            # if len(z) < 64:
            #     continue
            imgs = generator(z, gen_labels)
        imgs_result = torch.cat((imgs_result, imgs),0) 

    N = len(imgs_result)
    # print(np.shape(imgs_result))
    dataloader_generated = torch.utils.data.DataLoader(imgs_result, batch_size=batch_size) 

    # Load inception model
    ## we need to load our pre-trained model
    ckpt_classifier = torch.load(path_classifier)
    model = model_dict[opt.model]().to(opt.device)
    model.load_state_dict(ckpt_classifier['model'])
    model.eval()

    def get_pred(x):
        x = model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 10)) # 10 classes

    for i, batch in enumerate(dataloader_generated, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        # print((i*batch_size,i*batch_size + batch_size_i))
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)  # ????????? model??? ????????? ????????? preds ????????? ?????????.

    # print(np.shape(preds))
    # Now compute the mean kl-div --> kl-div??? ???????????? ??? ????????? ??????????????? ???.
    split_scores = []

    # ????????? ?????? split=1 ??????.
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0) # p(y) ??????
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]    # p(y|x)
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)


# ?????? import?????? ?????? ?????????
if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    mnist = dset.MNIST(root='./data/mnist', download=False,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize(([0.5], [0.5]), ([0.5], [0.5]))
                             ])
    )

    print ("Calculating Inception Score...")
    print (inception_score(opt.generator_path, opt.ckpt_dir, cuda=True, batch_size=64, resize=True, splits=10))
    # ??? img ????????? ?????? mnist data??? ??????? Generated img??? ???????????????.