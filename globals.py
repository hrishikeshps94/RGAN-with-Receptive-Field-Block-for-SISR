SCALE_FACTOR = 4
TRAINING_CROP_SIZE = 256
TRAINING_BATCH_SIZE = 2
TRAINING_LEARNING_RATE = 0.001
GAN_TRAINING_LEARNING_RATE = 0.0001
USE_AMP = True
# Feature extraction layer parameter configuration
feature_model_extractor_node = "features.34"
feature_model_normalize_mean = [0.485, 0.456, 0.406]
feature_model_normalize_std = [0.229, 0.224, 0.225]
l1_loss_weight = 10
vgg_loss_weight = 1
adv_loss_weight = 5e-3