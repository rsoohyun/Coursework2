import os, argparse
from torch._C import device
from tqdm import tqdm
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models import model_dict
from DCGAN_2 import Generator as DCGAN_G
from CGAN import Generator as CGAN_G
from DCGAN_2_vbn import Generator as DCGAN_vbn_G
from CGAN_vbn import Generator as CGAN_vbn_G

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--model", type=str, default="vgg8", help="model name")
parser.add_argument("--confidence", type=int, default=0.7, help="confidence score for training classifier with generated images")

parser.add_argument("--generated", type=bool, default=False, help="use generated image or not")
parser.add_argument("--generator_path", type=str, default="", help="pretrained generator")
parser.add_argument("--gan_model", type=str, default="DCGAN_2", choices=["DCGAN_2", "DCGAN_2_vbn", "CGAN", "CGAN_vbn"], help="kind of generator that we will use to generate image")
parser.add_argument("--trial", type=int, default=1, help="-th trial")
parser.add_argument("--ckpt_dir", type=str, default="./classifier_ckpt", help="checkpoint path of trained classifier")
parser.add_argument("--ckpt_path", type=str, default=None, help="Resume training from this checkpoint")

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available():
    opt.device = torch.device("cuda")
    opt.Tensor = torch.cuda.FloatTensor
    opt.LongTensor = torch.cuda.LongTensor
else:
    opt.device = torch.device("cpu")
    opt.Tensor = torch.FloatTensor
    opt.LongTensor = torch.LongTensor

opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

# data loading
os.makedirs("./data/mnist", exist_ok=True)
transform = transforms.Compose(
            [transforms.Resize(opt.img_size), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])]
            )

train_dataloader = DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transform,
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=2
)
test_dataloader = DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=False,
        download=True,
        transform=transform,
    ),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=2
)


def valid(opt, global_step):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    
    for step, data in enumerate(test_dataloader):
        inputs, labels = data
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)
        
        with torch.no_grad():
            logits = model(inputs)

            eval_loss = criterion(logits, labels)

            _, preds = logits.max(1)
            
            test_loss += eval_loss.item()
            correct += preds.eq(labels).sum()
    
    test_loss = test_loss / len(test_dataloader)
    accuracy = correct.float() / len(test_dataloader.dataset)
    # print('\nEval) global step: {}, Average loss: {:.4f}, Eval Accuracy: {:.4f}'.format(
    #     global_step, 
    #     test_loss,
    #     accuracy))

    return accuracy, test_loss


def train(opt, best):
    global_step, best_acc = 0, best

    train_loss = []
    train_step = []

    test_step = []
    test_loss = []
    test_acc = []

    # load generator
    if opt.generated:
        if opt.gan_model == "DCGAN_2":
            generator = DCGAN_G(opt)
        elif opt.gan_model == "DCGAN_2_vbn":
            generator = DCGAN_vbn_G(opt)
        elif opt.gan_model == "CGAN":
            generator = CGAN_G(opt)
        elif opt.gan_model == "CGAN_vbn":
            generator = CGAN_vbn_G(opt)
        
        ckpt_G = torch.load(opt.generator_path)
        try:
            generator.load_state_dict(ckpt_G['generator'])
        except RuntimeError as e:
            print('wrong checkpoint')
        else:    
            print('generator checkpoint is loaded !')
    
        generator.to(opt.device)

    for epoch in range(opt.n_epochs):
        model.train()
        epoch_iterator = tqdm(train_dataloader,
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        
        for step, data in enumerate(epoch_iterator):
            inputs, labels = data
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)

            if opt.generated:
                z = Variable(opt.Tensor(np.random.normal(0, 1, (inputs.shape[0], opt.latent_dim))))
                gen_labels = Variable(opt.LongTensor(np.random.randint(0, opt.n_classes, opt.batch_size)))

                if (opt.gan_model == "DCGAN_2") or (opt.gan_model == "DCGAN_2_vbn"): 
                    with torch.no_grad():
                        gen_imgs = generator(z)
                else:
                    with torch.no_grad():
                        gen_imgs = generator(z, gen_labels)
                
                # https://towardsdatascience.com/artificial-data-for-image-classification-5b2ede40640f
                logits = model(gen_imgs)
                pred_labels = torch.argmax(logits,1)
                confidence = opt.confidence

                prob = F.softmax(logits, dim=1)
                mostlikely = np.asarray([prob[i, pred_labels[i]].item() for i in range(len(prob))])
                want_keep = mostlikely > confidence
                weight = len(prob) / len(want_keep)
                loss = 1 * weight
                if sum(want_keep) != 0:
                    loss = criterion(logits[want_keep], pred_labels[want_keep]) * weight
                
                loss.backward()

            else:
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            #scheduler.step()
            global_step += 1

            if (global_step % 200 == 0):
                epoch_iterator.set_description(
                    "Training (%d Epoch / %d Iteration) (loss=%2.5f)" % ((epoch+1), global_step, loss.item())
                )
            train_loss.append(loss.item())
            train_step.append(global_step)

            accuracy, test_loss_v = valid(opt, global_step)
            test_acc.append(accuracy)
            test_loss.append(test_loss_v)
            test_step.append(global_step)
        
            if best_acc < accuracy:
                best_acc = accuracy
                ckpt_path = os.path.join(opt.ckpt_dir, f"{opt.model}_trial{opt.trial}_best.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    'best_acc': best_acc
                }, ckpt_path)
            
            if (global_step % 200 == 0):
                print('\nEval) global step: {}, Average loss: {:.4f}, Eval Accuracy: {:.4f}'.format(
                    global_step, 
                    test_loss_v,
                    accuracy))
            
            model.train()

    print("Best Accuracy: ", best_acc)

    return train_step, train_loss, test_step, test_loss, test_acc


# Model & optimizer
model = model_dict[opt.model]().to(opt.device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
criterion = nn.CrossEntropyLoss()

# ckpt setting
os.makedirs(opt.ckpt_dir, exist_ok=True)

best_acc = 0
if opt.ckpt_path is not None:
    ckpt = torch.load(opt.ckpt_path)
    try:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_acc = ckpt['best_acc']
    except RuntimeError as e:
        print('wrong checkpoint')
    else:    
        print('checkpoint is loaded !')
        print('current best accuracy : %.2f' % best_acc)


train_step, train_loss, test_step, test_loss, test_acc = train(opt, best_acc)

fig_recon, ax_recon = plt.subplots(2,1)
ax_recon[0].plot(train_step, train_loss)
ax_recon[0].set_xlabel("global step")
ax_recon[0].set_ylabel("train loss")
ax_recon[0].set_title("Train/loss")

ax_recon[1].plot(test_step, test_loss)
ax_recon[1].set_xlabel("global step")
ax_recon[1].set_ylabel("test loss")
ax_recon[1].set_title("Test/loss")

# ax_recon[2].plot(test_step, test_acc)
# ax_recon[2].set_xlabel("global step")
# ax_recon[2].set_ylabel("test accuracy")
# ax_recon[2].set_title("Test/Accuracy")

fig_recon.tight_layout(pad=2.0)

plt.savefig(f'{opt.model}_Trial{opt.trial}.png')