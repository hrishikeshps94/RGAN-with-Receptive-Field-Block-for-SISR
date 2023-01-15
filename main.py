import os
import argparse
from datetime import datetime
from globals import SCALE_FACTOR
def main(args):
    if args.network_type=='GAN':
        from train_GAN import Train as gen_train
        gen_train(args)
    elif args.network_type=='generator':
        from train_gen import Train as gen_train
        gen_train(args)
    else:
        print("select a valid mode network type generator/GAN")
    return None


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Get command-line arguments.")
    arg_parser.add_argument('--mode', type=str, choices=['test', 'train'], default='train')

    main_path = os.path.dirname(os.path.abspath(__file__))
    # print(main_path)
    # ------------------------------------------------
    # Testing arguments
    # ------------------------------------------------

    # Input folder for test images
    # arg_parser.add_argument('--input', type=str, default='/media/hrishi/data/WORK/FYND/super_resolution/dataset/DIV2K_valid_LR_bicubic_X2/DIV2K_valid_LR_bicubic/X2/')
    # Output folder for test results
    arg_parser.add_argument('--output', type=str, default=os.path.join(main_path, 'output'))

    # Patch size for patch-based testing of large images.
    # Make sure the patch size is small enough that your GPU memory is sufficient.
    # arg_parser.add_argument('--patch_size', type=int, default=128)
    current_time = datetime.now()
    # Checkpoint folder that contains the generator.pth and discriminator.pth checkpoint files.
    arg_parser.add_argument('--checkpoint_folder', type=str,
                            default=os.path.join(main_path, 'checkpoints', 'x' + str(SCALE_FACTOR) + '__sr__'+'acer_l1_fft'))

    # ------------------------------------------------
    # Training arguments
    # ------------------------------------------------

    # Log folder where Tensorboard logs are saved
    arg_parser.add_argument('--log_name', type=str,
                            default=str(SCALE_FACTOR) + '__sr__'+'normal')
    arg_parser.add_argument('--epochs', type=int,
                            default=100)
    arg_parser.add_argument('--resume',action='store_true')

    # Folders for training and validation datasets.
    arg_parser.add_argument('--train_input', type=str, default=os.path.join('test','HR'))
    arg_parser.add_argument('--valid_input', type=str, default=os.path.join('test','HR'))
    # Define whether we use only the generator or the whole pipeline with the discriminator for training.
    arg_parser.add_argument('--network_type', type=str, choices=['generator', 'GAN'], default='GAN')

    arg_list = arg_parser.parse_args()
    main(arg_list)