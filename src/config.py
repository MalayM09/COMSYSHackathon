import os
import numpy as np
import torch

class OptimizedArcFaceConfig:
    """
    Research-optimized configuration for ConvNeXt + ArcFace hybrid model.
    """

    # Dataset paths
    DATASET_PATH = "/kaggle/input/facecom/Comys_Hackathon5"
    TASK_A_TRAIN_PATH = os.path.join(DATASET_PATH, "Task_A/train")
    TASK_A_VAL_PATH = os.path.join(DATASET_PATH, "Task_A/val")
    TASK_B_TRAIN_PATH = os.path.join(DATASET_PATH, "Task_B/train")
    TASK_B_VAL_PATH = os.path.join(DATASET_PATH, "Task_B/val")

    # Model architecture
    IMG_SIZE = 224
    CHANNELS = 3
    BACKBONE = "convnext_tiny"
    PRETRAINED = True
    EMBEDDING_DIM = 512
    GENDER_CLASSES = 2

    # Batch and training
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2

    # Training phases
    NUM_EPOCHS_VERIFICATION = 40
    NUM_EPOCHS_GENDER = 15
    NUM_EPOCHS_JOINT = 8

    # Learning rates
    VERIFICATION_LR_BACKBONE = 1e-4
    VERIFICATION_LR_HEAD = 3e-4
    VERIFICATION_LR_ARCFACE = 3e-4
    GENDER_LR_HEAD = 5e-5
    JOINT_LR = 5e-6

    # Regularization
    VERIFICATION_DROPOUT = 0.4
    GENDER_DROPOUT = 0.5
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTHING = 0.1

    # ArcFace parameters
    ARCFACE_MARGIN = 0.5
    ARCFACE_SCALE = 64
    NUM_IDENTITIES = 500

    # Loss weights
    VERIFICATION_LOSS_WEIGHT = 1.0
    ARCFACE_LOSS_WEIGHT = 1.0
    GENDER_LOSS_WEIGHT = 1.0
    TRIPLET_LOSS_WEIGHT = 0.2

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    PIN_MEMORY = True
    USE_AMP = True
    SEED = 42
    CHECKPOINT_DIR = "./models"  # Change as needed for local/colab

    # Thresholds
    SIMILARITY_THRESHOLDS = np.arange(0.1, 0.95, 0.02)

    # Augmentation
    USE_ALBUMENTATIONS = True
    ROTATION_LIMIT = 15
    BRIGHTNESS_LIMIT = 0.15
    CONTRAST_LIMIT = 0.15
    PERSPECTIVE_PROB = 0.3
    ELASTIC_PROB = 0.2
    GAUSSIAN_NOISE_PROB = 0.3
    BLUR_PROB = 0.2

    # Advanced training
    PATIENCE = 15
    MIN_DELTA = 1e-4
    MAX_GRAD_NORM = 0.3
    CLEAR_CACHE_FREQUENCY = 5

    # Task B settings
    MAX_IDENTITIES_TRAIN = 500
    MAX_IDENTITIES_VAL = 200
    PAIRS_PER_IDENTITY = 25
    HARD_NEGATIVE_RATIO = 0.4

    # Temperature scaling
    USE_TEMPERATURE_SCALING = True
    INITIAL_TEMPERATURE = 1.5

    # Curriculum learning
    USE_CURRICULUM_LEARNING = True
    CURRICULUM_START_EPOCH = 10

    @classmethod
    def create_optimizers(cls, model):
        """
        Create optimizers for verification, gender, and joint training.
        """
        verification_optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': cls.VERIFICATION_LR_BACKBONE},
            {'params': model.verification_head.parameters(), 'lr': cls.VERIFICATION_LR_HEAD},
            {'params': model.arcface_loss.parameters(), 'lr': cls.VERIFICATION_LR_ARCFACE}
        ], weight_decay=cls.WEIGHT_DECAY, eps=1e-8)

        gender_optimizer = torch.optim.AdamW(
            model.gender_head.parameters(),
            lr=cls.GENDER_LR_HEAD,
            weight_decay=cls.WEIGHT_DECAY,
            eps=1e-8
        )

        joint_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cls.JOINT_LR,
            weight_decay=cls.WEIGHT_DECAY,
            eps=1e-8
        )

        return verification_optimizer, gender_optimizer, joint_optimizer

    @classmethod
    def create_schedulers(cls, optimizers, total_epochs):
        """
        Create learning rate schedulers for each optimizer.
        """
        verification_optimizer, gender_optimizer, joint_optimizer = optimizers

        verification_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            verification_optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7
        )

        gender_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            gender_optimizer,
            T_max=cls.NUM_EPOCHS_GENDER,
            eta_min=1e-8
        )

        joint_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            joint_optimizer,
            gamma=0.95
        )

        return verification_scheduler, gender_scheduler, joint_scheduler
