from torch import nn
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime
import torch
from torch.autograd import Variable
import glob,tqdm,cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

class Generator(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()        
        # 今回は転移学習を行わない
        model_vgg16=models.vgg16(pretrained=True) #SSL errorを防ぐ, 左記linkからdownload
        self.encoder_first = model_vgg16.features[:17] # 重み固定して使う部分
        self.encoder_last = model_vgg16.features[17:-1] # 学習する部分
        self.decoder = nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=1), 
                    nn.LeakyReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), 
                    nn.LeakyReLU(),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(64, 1, 1, padding=0),
                    nn.Sigmoid())

    def forward(self, x):
        x = self.encoder_first(x)
        x = self.encoder_last(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                    nn.Conv2d(4, 3, 1, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2))
        self.classifier = nn.Sequential(
                    nn.Linear(64*32*24, 100, bias=True),
                    nn.Tanh(),
                    nn.Linear(100, 2, bias=True),
                    nn.Tanh(),
                    nn.Linear(2, 1, bias=True),
                    nn.Sigmoid())

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
class SALICONImgDataSet(Dataset):
    def __init__(self, img_dir,img_fnames, img_tfms, mask_dir, mask_tfms):
        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.img_transform = img_tfms
        self.mask_dir = mask_dir
        self.mask_transform = mask_tfms
    def __getitem__(self, i):
        fname = self.img_fnames[i]
        mask_fname = fname.split(".")[0]+".png"
        fpath = os.path.join(self.img_dir, fname)
        img = cv2.imread(fpath)
        mpath = os.path.join(self.mask_dir, mask_fname)
        mask = cv2.imread(mpath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # transform
        img_ = np.asarray(img)
        img_ = img_[:,:,:3]
        ch = img_.shape[2]
        if ch!=3:
          img = img.convert('RGB')
        img = self.img_transform(Image.fromarray(img))
        mask = self.mask_transform(Image.fromarray(mask))
        return img, mask
    def __len__(self):
        return len(self.img_fnames)
    
    
def train(root_dataset_dir):
    #-----------------
    # SETTING
    img_dir_train = os.path.join(root_dataset_dir,'images','train')
    target_dir_train = os.path.join(root_dataset_dir,'graymaps','train')
    img_dir_val = os.path.join(root_dataset_dir,'images','val')
    target_dir_val = os.path.join(root_dataset_dir,'graymaps','val')
    train_img_ids = [path.name for path in Path(img_dir_train).glob('*.jpg')]
    val_img_ids = [path.name for path in Path(img_dir_val).glob('*.jpg')]
    img_tfms = transforms.Compose([transforms.Resize((192, 256)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    mask_tfms = transforms.Compose([transforms.Resize((192, 256)), transforms.ToTensor()])
    alpha = 0.005 # Generatorの損失関数のハイパーパラメータ。論文の推奨値は0.005
    epochs = 120
    batch_size = 32 # 論文では32
    #-----------------

    # 開始時間をファイル名に利用
    start_time_stamp = '{0:%Y%m%d-%H%M%S}'.format(datetime.now())

    save_dir = "./log/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # データローダーの読み込み
    train_dataset = SALICONImgDataSet(
                        img_dir=img_dir_train,img_fnames=train_img_ids, img_tfms=img_tfms, mask_dir=target_dir_train, mask_tfms=mask_tfms
                    )
    val_dataset = SALICONImgDataSet(
                        img_dir=img_dir_val,img_fnames=val_img_ids, img_tfms=img_tfms, mask_dir=target_dir_val, mask_tfms=mask_tfms
                    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 1, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers = 1, pin_memory=True, sampler=None)
    print("dataloader is prepared")
    # モデルと損失関数の読み込み
    loss_func = torch.nn.BCELoss().to(DEVICE)
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # 最適化手法の定義（論文中の設定を使用）
    optimizer_G = torch.optim.Adagrad([
                    #{'params': generator.encoder_first.parameters()},
                    {'params': generator.encoder_last.parameters()},
                    {'params': generator.decoder.parameters()}
                ], lr=0.0001, weight_decay=3*0.0001)
    optimizer_D = torch.optim.Adagrad(discriminator.parameters(), lr=0.0001, weight_decay=3*0.0001)

    # 学習
    losses_d = []
    losses_g = []
    for epoch in range(epochs):
        n_updates = 0 # イテレーションのカウント
        n_discriminator_updates = 0
        n_generator_updates = 0
        d_loss_sum = 0
        g_loss_sum = 0
        with tqdm.tqdm(total=len(train_loader),unit="batch") as pbar:
            pbar.set_description(f"Epoch[{epoch+1}/{epochs}]")
            for i, data in enumerate(train_loader):  
            #for i, data in enumerate(train_loader):
            imgs = data[0] # ([batch_size, rgb, h, w])
            salmaps = data[1] # ([batch_size, 1, h, w])
            # Discriminator用のラベルを作成
            valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False).to(DEVICE)
            fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False).to(DEVICE)

            imgs = Variable(imgs).to(DEVICE)
            real_salmaps = Variable(salmaps).to(DEVICE)

            # イテレーションごとにGeneratorとDiscriminatorを交互に学習
            if n_updates % 2 == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
                gen_salmaps = generator(imgs)

                # Discriminatorへの入力用に元の画像と生成したSaliency Mapを結合して4チャンネル(=rgb(=3channel)+saliecy(=1channel))の配列を作る
                fake_d_input = torch.cat((imgs, gen_salmaps.detach()), 1) # ([batch_size, rgb(=3channel)+saliecy(=1channel), h, w])

                # Generatorの損失関数を計算
                g_loss1 = loss_func(gen_salmaps, real_salmaps)
                g_loss2 = loss_func(discriminator(fake_d_input), valid)
                g_loss = alpha*g_loss1 + g_loss2

                g_loss.backward()
                optimizer_G.step()

                g_loss_sum += g_loss.item()
                n_generator_updates += 1

            else:
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Discriminatorへの入力用に元の画像と正解データのSaliency Mapを結合して4チャンネルの配列を作る            
                real_d_input = torch.cat((imgs, real_salmaps), 1) # ([batch_size, rgbs, h, w])

                # Discriminatorの損失関数を計算
                real_loss = loss_func(discriminator(real_d_input), valid)
                fake_loss = loss_func(discriminator(fake_d_input), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                d_loss_sum += d_loss.item()
                n_discriminator_updates += 1

            n_updates += 1
            if n_discriminator_updates>0:
                loss_d = d_loss_sum/n_discriminator_updates
            else:
                loss_d = d_loss_sum
            pbar.set_postfix({"loss D":loss_d,"loss G":g_loss_sum/n_generator_updates })
            pbar.update(1)
        # 重みの保存
        # 5エポックごとと、最後のエポックを保存する
        if ((epoch+1)%5==0)or(epoch==epochs-1):
            generator_save_path = '{}.pkl'.format(os.path.join(save_dir, "{}_generator_epoch{}".format(start_time_stamp, epoch)))
            discriminator_save_path = '{}.pkl'.format(os.path.join(save_dir, "{}_discriminator_epoch{}".format(start_time_stamp, epoch)))
            torch.save(generator.state_dict(), generator_save_path)
            torch.save(discriminator.state_dict(), discriminator_save_path)
        # エポックごとにlossを格納
        losses_d.append(d_loss_sum/n_discriminator_updates)
        losses_g.append(g_loss_sum/n_generator_updates)
        with open(save_dir+"result.txt", mode="a") as f:
            f.write(str(loss_d)+",")
            f.write(str(g_loss_sum/n_generator_updates)+"\n")
        # エポックごとにValidationデータの一部を可視化
        with torch.no_grad():
            print("validation")
            for i, data in enumerate(val_loader):
                image = Variable(data[0]).to(DEVICE)
                gen_salmap = generator(imgs)
                gen_salmap_np = np.array(gen_salmaps.data.cpu())[0, 0]
                gen_salmap_np = gen_salmap_np*255
                gen_salmap_np_int = gen_salmap_np.astype(int)
                fig = plt.figure(figsize=(8,20))
                ax1 = fig.add_subplot(3, 1, 1)
                img_bgr=np.array(image[0].cpu()).transpose(1, 2, 0)
                img_bgr*=255
                img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
                ax1.imshow(img_rgb.astype(int))
                # bottom
                ax2 = fig.add_subplot(3, 1, 2)
                ax2.imshow(gen_salmap_np_int)
                ax3 = fig.add_subplot(3, 1, 3)
                sal_color=cv2.cvtColor(gen_salmap_np,cv2.COLOR_GRAY2RGB)
                img_merge=cv2.addWeighted(255-sal_color,1,img_bgr,1,0)
                img_merge_=cv2.cvtColor(img_merge,cv2.COLOR_BGR2RGB)
                img_merge_=img_merge_.astype(int)
                ax3.imshow(img_merge_)
                # show plots
                fig.tight_layout()
                fig.savefig(save_dir+str(epoch+1)+".png")
                if i==0:
                    break
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    root_dataset_dir,img_dir,map_dir
    parser.add_argument('-dataset_dir', default="data", type=str)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    train(dataset_dir)