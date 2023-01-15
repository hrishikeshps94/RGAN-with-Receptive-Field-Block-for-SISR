import os
import torch
import torchvision
from torch.utils.data import Dataset
import PIL
from globals import TRAINING_CROP_SIZE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device used = {device}')

class TrainDataset(Dataset):
    def __init__(self, folder):
        self.image_files = []
        for dir_path, _, file_names in os.walk(folder):
            for f in file_names:
                file_name = os.path.join(dir_path, f)
                self.image_files.append(file_name)
        self.image_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(TRAINING_CROP_SIZE),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip()
        ])
        self.tensor_convert = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize(size=(TRAINING_CROP_SIZE//4,TRAINING_CROP_SIZE//4),interpolation=\
            torchvision.transforms.InterpolationMode.BICUBIC,antialias=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        img = PIL.Image.open(image_file).convert('RGB')
        hr_img = self.image_transform(img)
        lr_img = self.resize(hr_img)
        hr_img,lr_img = self.tensor_convert(hr_img),self.tensor_convert(lr_img)
        return {'lowres_img': lr_img,
        'ground_truth_img': hr_img
        }

class ValidDataset(Dataset):
    def __init__(self, folder):
        self.image_files = []
        for dir_path, _, file_names in os.walk(folder):
            for f in file_names:
                file_name = os.path.join(dir_path, f)
                self.image_files.append(file_name)
        self.tensor_convert = torchvision.transforms.ToTensor()
        # self.resize = torchvision.transforms.Resize(size=(TRAINING_CROP_SIZE//4,TRAINING_CROP_SIZE//4),interpolation=\
        #     torchvision.transforms.InterpolationMode.BICUBIC,antialias=True,)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        hr_img = PIL.Image.open(image_file).convert('RGB')
        w,h = hr_img.size
        assert (h%4 == 0)
        assert (w%4 == 0)
        lr_img = torchvision.transforms.functional.resize(hr_img,(h//4,w//4),interpolation=\
        torchvision.transforms.InterpolationMode.BICUBIC,antialias=True)
        hr_img,lr_img = self.tensor_convert(hr_img),self.tensor_convert(lr_img)
        return {'lowres_img': lr_img,
        'ground_truth_img': hr_img
        }


class TestDataset(Dataset):
    def __init__(self, folder):
        self.image_files = []
        for dir_path, _, file_names in os.walk(folder):
            for f in file_names:
                file_name = os.path.join(dir_path, f)
                self.image_files.append(file_name)
        self.tensor_convert = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        hr_img = PIL.Image.open(image_file).convert('RGB')
        w,h = hr_img.size
        assert (h%4 == 0)
        assert (w%4 == 0)
        lr_img = torchvision.transforms.functional.resize(hr_img,(h//4,w//4),interpolation=\
        torchvision.transforms.InterpolationMode.BICUBIC,antialias=True)
        hr_img,lr_img = self.tensor_convert(hr_img),self.tensor_convert(lr_img)
        return {'lowres_img': lr_img,
        'ground_truth_img': hr_img,
        'img_name': os.path.basename(image_file)
        }


# class TestDataset(Dataset):
#     def __init__(self, folder):
#         self.image_files = []
#         for dir_path, _, file_names in os.walk(folder):
#             for f in file_names:
#                 print(f)
#                 file_name = os.path.join(dir_path, f)
#                 self.image_files.append(file_name)
#         self.tensor_convert = torchvision.transforms.ToTensor()

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, item):
#         image_file = self.image_files[item]
#         img = PIL.Image.open(image_file).convert('RGB')

#         # If height > width, we flip the image
#         flipped = False
#         resize = False
#         if img.height > img.width:
#             img = img.transpose(PIL.Image.TRANSPOSE)
#             flipped = True
#         org_h,org_w = img.size

#         img = self.tensor_convert(img)
#         return {'lowres_img': img,
#             'img_name': os.path.basename(image_file),
#             'flipped': flipped,
#             'resize': resize,
#             'org_shape':(org_h,org_w)
#         }
