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
from model import Generator
from globals import TRAINING_BATCH_SIZE,TRAINING_LEARNING_RATE,TRAINING_CROP_SIZE
from torchinfo import summary
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
    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=TRAINING_LEARNING_RATE)
        self.l1_loss = torch.nn.L1Loss()
        # self.val_l1_loss = torch.nn.L1Loss()
        # self.vgg_loss = VGG19PerceptualLoss()
    def init_lr_scheduler(self):
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
        self.train_generator()
    def save_generator_checkpoint(self,type='last'):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder, f'{self.args.network_type}')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder, f'{type}.pth')
        save_data = {
            'step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler_steplr.state_dict(),
            'best_psnr':self.max_psnr,'best_lpips':self.min_lpips,
        }
        torch.save(save_data, checkpoint_filename)
    def load_generator_checkpoint_for_training(self):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder, f'{self.args.network_type}')
        checkpoint_filename = os.path.join(checkpoint_folder, 'best.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(checkpoint_filename)
        self.global_step = data['step']
        self.generator.load_state_dict(data['generator_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.scheduler_steplr.load_state_dict(data['scheduler_state_dict'])
        self.max_psnr,self.min_lpips = data['best_psnr'],data['best_lpips']
        print(f"Restored model at step {self.global_step}.")


    def train_generator(self):
        if self.args.resume:
            self.load_generator_checkpoint_for_training()
        max_step = self.max_epoch * len(self.train_dataloader)
        steps_per_epoch = len(self.train_dataloader)
        print(f'Training started from step = {self.global_step}/{max_step}')
        while self.global_step < max_step:
            self.generator_training_epoch()
            if ((self.global_step%((steps_per_epoch)*2)==0) and (self.global_step!=0)):
                self.generator_validation_epoch()


    def generator_training_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
        steps_per_epoch = len(self.train_dataloader)
        self.generator.train()
        for batch in tqdm(self.train_dataloader):
            self.global_step += 1
            self.optimizer.zero_grad()
            lowres_img = batch['lowres_img'].to(device)
            ground_truth = batch['ground_truth_img'].to(device)

            generator_output = self.generator(lowres_img)
            loss = self.l1_loss(generator_output, ground_truth)
            loss.backward()
            self.optimizer.step()
            self.scheduler_steplr.step()

            accumulate_loss += loss.item()
            accumulate_steps += 1

            if self.global_step % self.steps_per_epoch == 0:
                print(f'Training step {self.global_step} -- L1 loss: {accumulate_loss / accumulate_steps}')
                wandb.log({'train_generator_l1_loss':accumulate_loss / accumulate_steps})
                accumulate_loss = 0
                accumulate_steps = 0
                wandb.log({'learningrate':self.optimizer.param_groups[0]['lr']})

            if self.global_step % steps_per_epoch == 0:
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