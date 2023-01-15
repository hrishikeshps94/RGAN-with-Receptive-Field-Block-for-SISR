import os,torch,math
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device used= {device}')
from globals import *
from dataset import TestDataset
from model import Generator
from torch.utils.data import DataLoader
import imageio
import tqdm


class Test():
    def __init__(self, args):
        self.args = args
        self.init_model_for_testing()
        self.restore_models_for_testing()
        self.init_test_data()
        self.launch_test()
    def init_model_for_testing(self):
        self.generator = Generator(input_dim=3,trunk_a_count=8,trunk_rfb_count=8)
        self.generator = self.generator.to(device)

    def init_test_data(self):
        test_folder = self.args.input
        print(os.listdir(test_folder))
        test_dataset = TestDataset(test_folder)
        self.test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=os.cpu_count())

    def restore_models_for_testing(self):
        checkpoint_folder = self.args.checkpoint_folder
        generator_checkpoint_filename = os.path.join(checkpoint_folder, f'{self.args.network_type}','best_PSNR.pth')
        if (not os.path.exists(generator_checkpoint_filename)):
            print("Error: could not locate network checkpoints. Make sure the files are in the right location.")
            print(f"The generator checkpoint should be at {generator_checkpoint_filename}")
            exit()
        data = torch.load(generator_checkpoint_filename,map_location='cpu')
        self.generator.load_state_dict(data['generator_state_dict'])

    def launch_test(self):
        for batch in tqdm.tqdm(self.test_dataloader):
            lowres_img = batch['lowres_img'].to(device)
            orgres_img = batch['ground_truth_img'].to(device)
            image_name = batch['img_name']
            print(lowres_img.shape)

            with torch.no_grad():
                highres_output = self.generator(lowres_img)
                highres_image = (highres_output[0,...]*255.0).permute(1, 2, 0).clamp(0.0, 255.0).type(torch.uint8).cpu().numpy()
                lowres_img = (lowres_img[0,...]*255.0).permute(1, 2, 0).clamp(0.0, 255.0).type(torch.uint8).cpu().numpy()
                orgres_img = (orgres_img[0,...]*255.0).permute(1, 2, 0).clamp(0.0, 255.0).type(torch.uint8).cpu().numpy()
                output_folder = os.path.join(self.args.output,f'{self.args.network_type}')
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_sr_image_name = str.split(image_name[0], '.')[0] + '_sr_.jpg'
                output_lr_image_name = str.split(image_name[0], '.')[0] + '_lr_.jpg'
                output_hr_image_name = str.split(image_name[0], '.')[0] + '_hr_.jpg'
                output_sr_file = os.path.join(output_folder, output_sr_image_name)
                output_lr_file = os.path.join(output_folder, output_lr_image_name)
                output_hr_file = os.path.join(output_folder, output_hr_image_name)
                imageio.imwrite(output_sr_file, highres_image)
                imageio.imwrite(output_lr_file, lowres_img)
                imageio.imwrite(output_hr_file, orgres_img)
                print(f"Saving output image at {output_sr_file} and {output_lr_file} .")

def main(args):
    Test(args)

if __name__ == '__main__':
    main_path = os.path.dirname(os.path.abspath(__file__))
    arg_parser = argparse.ArgumentParser(description="Get command-line arguments.")
    arg_parser.add_argument('--input', type=str, default='test/HR')
    arg_parser.add_argument('--output', type=str, default=os.path.join(main_path, 'output'))
    arg_parser.add_argument('--checkpoint_folder', type=str,
                            default=os.path.join(main_path, 'checkpoints','SR'))
    arg_parser.add_argument('--network_type', type=str, choices=['generator', 'GAN'], default='generator')

    arg_list = arg_parser.parse_args()
    main(arg_list)