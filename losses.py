import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F

class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(self, feature_model_extractor_node: str,
                 feature_model_normalize_mean: list,
                 feature_model_normalize_std: list) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(True)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        content_loss = F.l1_loss(sr_feature, hr_feature)

        return content_loss