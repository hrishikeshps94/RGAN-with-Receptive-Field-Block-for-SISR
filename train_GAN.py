import os
# import math
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TrainDataset, ValidDataset
from globals import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from model import Generator,Discriminator
from globals import TRAINING_BATCH_SIZE,TRAINING_LEARNING_RATE,TRAINING_CROP_SIZE,feature_model_extractor_node,\
    feature_model_normalize_mean,feature_model_normalize_std,l1_loss_weight,adv_loss_weight,vgg_loss_weight,GAN_TRAINING_LEARNING_RATE
from torchinfo import summary
from losses import ContentLoss
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

class Train():
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.max_epoch = args.epochs
        self.max_psnr = 0
        self.min_lpips = float('inf')
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize =True).to(device)
        if self.args.mode == 'train':
            self.init_model_for_training()
            self.init_train_data()
            self.init_summary()
            self.init_optimizer()
            self.init_lr_scheduler()
            self.steps_per_epoch = len(self.train_dataloader)
            self.launch_training()
        else:
            print('Testing has not been Integrated :p')
    def init_summary(self):
        wandb.init(project="NewSR",name=self.args.log_name,mode="offline")

    def init_model_for_training(self):
        self.generator = Generator(input_dim=3,trunk_a_count=8,trunk_rfb_count=8)
        summary(self.generator, input_size=(TRAINING_BATCH_SIZE, 3, TRAINING_CROP_SIZE//4, TRAINING_CROP_SIZE//4))
        self.generator = self.generator.to(device)
        if self.args.network_type == "GAN":
            checkpoint_filename = os.path.join(self.args.checkpoint_folder,'generator','best.pth')
            assert os.path.exists(checkpoint_filename)
            data = torch.load(checkpoint_filename)
            self.generator.load_state_dict(data['generator_state_dict'])
            self.discriminator = Discriminator()
            summary(self.discriminator, input_size=(TRAINING_BATCH_SIZE, 3, TRAINING_CROP_SIZE, TRAINING_CROP_SIZE))
            self.discriminator = self.discriminator.to(device)

    def init_optimizer(self):
        if self.args.network_type == "GAN":
            self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=GAN_TRAINING_LEARNING_RATE)
            self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=GAN_TRAINING_LEARNING_RATE)
            self.vgg_loss = ContentLoss(feature_model_extractor_node,feature_model_normalize_mean,feature_model_normalize_std).to(device)
            self.l1_loss = torch.nn.L1Loss().to(device)
            self.BCE = torch.nn.BCEWithLogitsLoss().to(device)
        elif self.args.network_type == "generator":
            self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=TRAINING_LEARNING_RATE)
            self.l1_loss = torch.nn.L1Loss().to(device)

    def init_lr_scheduler(self):
        if self.args.network_type == "GAN":
            total_steps = self.max_epoch * (len(self.train_dataloader))
            self.scheduler_steplr = CosineAnnealingLR(self.optimizer,total_steps)
            self.d_scheduler_steplr = CosineAnnealingLR(self.d_optimizer,total_steps)
        elif self.args.network_type == "generator":
            total_steps = self.max_epoch * (len(self.train_dataloader))
            self.scheduler_steplr = CosineAnnealingLR(self.optimizer,total_steps)

    def init_train_data(self):
        batch_size = TRAINING_BATCH_SIZE
        train_folder = self.args.train_input
        train_dataset = TrainDataset(train_folder)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=os.cpu_count())

        valid_folder = self.args.valid_input
        valid_dataset = ValidDataset(valid_folder)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=1,shuffle=True,num_workers=os.cpu_count())
    def launch_training(self):
        self.train_GAN()
    def save_generator_checkpoint(self,type='last'):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder, f'{self.args.network_type}')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder,f'{type}.pth')
        if self.args.network_type == "GAN":
            save_data = {
                'step': self.global_step,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler_steplr.state_dict(),
                'd_scheduler_state_dict': self.d_scheduler_steplr.state_dict(),
                'best_psnr':self.max_psnr,'best_lpips':self.min_lpips,
            }
        elif self.args.network_type == "generator":
            save_data = {
                'step': self.global_step,
                'generator_state_dict': self.generator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler_steplr.state_dict(),
                'best_psnr':self.max_psnr,'best_lpips':self.min_lpips,
            }
        torch.save(save_data, checkpoint_filename)
    def load_generator_checkpoint_for_training(self):
        checkpoint_folder = self.args.checkpoint_folder
        checkpoint_filename = os.path.join(checkpoint_folder,f'{self.args.network_type}','best.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(checkpoint_filename)
        self.global_step = data['step']
        self.generator.load_state_dict(data['generator_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.scheduler_steplr.load_state_dict(data['scheduler_state_dict'])
        self.max_psnr,self.min_lpips = data['best_psnr'],data['best_lpips']
        if self.args.network_type == "GAN":
            self.discriminator.load_state_dict(data['discriminator_state_dict'])
            self.d_optimizer.load_state_dict(data['d_optimizer_state_dict'])
            self.d_scheduler_steplr.load_state_dict(data['d_scheduler_state_dict'])
        print(f"Restored model at step {self.global_step}.")


    def train_GAN(self):
        if self.args.resume:
            self.load_generator_checkpoint_for_training()
        max_step = self.max_epoch * len(self.train_dataloader)
        steps_per_epoch = len(self.train_dataloader)
        print(f'Training started from step = {self.global_step}/{max_step}')
        while self.global_step < max_step:
            self.generator_training_epoch()
            if ((self.global_step%((steps_per_epoch)*1)==0) and (self.global_step!=0)):
                self.generator_validation_epoch()


    def generator_training_epoch(self):
        g_accumulate_loss = 0
        d_accumulate_loss = 0
        accumulate_steps = 0
        # steps_per_epoch = len(self.train_dataloader)
        self.generator.train()
        self.discriminator.train()
        for batch in tqdm(self.train_dataloader):
            self.global_step += 1
            self.d_optimizer.zero_grad()
            self.discriminator.zero_grad()
            lowres_img = batch['lowres_img'].to(device)
            ground_truth = batch['ground_truth_img'].to(device)
            # Set the real sample label to 1, and the false sample label to 0
            real_label = torch.full([lowres_img.size(0), 1], 1.0, dtype=lowres_img.dtype, device=device)
            fake_label = torch.full([lowres_img.size(0), 1], 0.0, dtype=lowres_img.dtype, device=device)
            generator_output = self.generator(lowres_img)
            hr_output = self.discriminator(ground_truth)
            sr_output = self.discriminator(generator_output.detach().clone())
            disc_loss = (self.BCE(hr_output - torch.mean(sr_output), real_label) + self.BCE(sr_output - torch.mean(hr_output), fake_label))/2
            disc_loss.backward()
            self.d_optimizer.step()
            self.d_scheduler_steplr.step()
            d_accumulate_loss += disc_loss.item()

            self.optimizer.zero_grad()
            self.generator.zero_grad()
            hr_output = self.discriminator(ground_truth.detach().clone())
            sr_output = self.discriminator(generator_output)
            l1_loss = self.l1_loss(generator_output, ground_truth)
            content_loss = self.vgg_loss(generator_output,ground_truth)
            gen_loss = (self.BCE(hr_output - torch.mean(sr_output), fake_label) + self.BCE(sr_output - torch.mean(hr_output), real_label))/2
            total_loss = l1_loss_weight*l1_loss + vgg_loss_weight*content_loss + adv_loss_weight*gen_loss
            total_loss.backward()
            self.optimizer.step()
            self.scheduler_steplr.step()

            g_accumulate_loss += total_loss.item()
            accumulate_steps += 1

            if self.global_step % self.steps_per_epoch == 0:
                print(f'Training step {self.global_step} -- G loss: {g_accumulate_loss / accumulate_steps}')
                print(f'Training step {self.global_step} -- D loss: {d_accumulate_loss / accumulate_steps}')
                wandb.log({'train_generator_loss':g_accumulate_loss / accumulate_steps})
                wandb.log({'train_discriminator_loss':d_accumulate_loss / accumulate_steps})
                d_accumulate_loss = 0
                g_accumulate_loss = 0
                accumulate_steps = 0
                wandb.log({'learningrate':self.optimizer.param_groups[0]['lr']})

            if self.global_step % self.steps_per_epoch == 0:
                print('Saving train model..........')
                self.save_generator_checkpoint(type='last')

    def generator_validation_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
        accumulate_psnr = 0
        accumulate_lpips = 0
        self.generator.eval()
        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader):
                lowres_img = batch['lowres_img'].to(device)
                ground_truth = batch['ground_truth_img'].to(device)

                generator_output = self.generator(lowres_img)
                loss = self.l1_loss(generator_output, ground_truth)
                step_psnr  = self.psnr(generator_output,ground_truth)
                step_lpips = self.lpips(generator_output,ground_truth)
                accumulate_loss += loss.item()
                accumulate_psnr += step_psnr.item()
                accumulate_lpips += step_lpips.item()
                accumulate_steps += 1

            print(f'Validation -- L1 loss: {accumulate_loss / accumulate_steps} --PSNR: {accumulate_psnr / accumulate_steps} \
               --LPIPS: {accumulate_lpips / accumulate_steps}')
            wandb.log({'val_generator_l1_loss':accumulate_loss / accumulate_steps})
            wandb.log({'val_generator_PSNR':accumulate_psnr / accumulate_steps})
            wandb.log({'val_ground_truth':wandb.Image(ground_truth[0].clamp(0.0, 1.0))})
            wandb.log({'val_generator_output':wandb.Image(generator_output[0].clamp(0.0, 1.0))})
            wandb.log({'val_lowres_img':wandb.Image(lowres_img[0].clamp(0.0, 1.0))})
            if (accumulate_psnr / accumulate_steps)>=self.max_psnr:
              self.max_psnr = accumulate_psnr / accumulate_steps
              print(f'Saving best train model of PSNR={self.max_psnr}')
              self.save_generator_checkpoint(type='best_PSNR')
            if (accumulate_lpips / accumulate_steps)<self.min_lpips:
              self.min_lpips = accumulate_lpips / accumulate_steps
              print(f'Saving best train model of LPIPS={self.min_lpips}')
              self.save_generator_checkpoint(type='best_LPIPS')