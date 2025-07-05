import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import OptimizedArcFaceConfig

class TaskADataset(Dataset):
    """Dataset for Task A: Gender Classification"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'male': 0, 'female': 1}
        self._load_data()

    def _load_data(self):
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir)
                               if f.lower().endswith(('.jpg', '.jpeg'))]
                for image_file in image_files:
                    image_path = os.path.join(class_dir, image_file)
                    self.image_paths.append(image_path)
                    self.labels.append(class_idx)
        print(f"TaskA {self.split}: Loaded {len(self.image_paths)} images "
              f"(Male: {self.labels.count(0)}, Female: {self.labels.count(1)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (OptimizedArcFaceConfig.IMG_SIZE, OptimizedArcFaceConfig.IMG_SIZE), (0, 0, 0))
        if self.transform:
            if OptimizedArcFaceConfig.USE_ALBUMENTATIONS:
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)
        return image, label

class TaskBDataset(Dataset):
    """Dataset for Task B: Face Verification"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.identity_data = {}
        self.pairs = []
        self.labels = []
        self.identity_labels = []
        self.identity_to_idx = {}
        self._load_identity_data()
        self._generate_pairs()

    def _load_identity_data(self):
        if not os.path.exists(self.root_dir):
            print(f"TaskB {self.split}: Root directory not found: {self.root_dir}")
            return
        identity_folders = [f for f in os.listdir(self.root_dir)
                            if os.path.isdir(os.path.join(self.root_dir, f))]
        max_identities = OptimizedArcFaceConfig.MAX_IDENTITIES_TRAIN if self.split == 'train' else OptimizedArcFaceConfig.MAX_IDENTITIES_VAL
        identity_folders = identity_folders[:max_identities]
        for idx, identity_folder in enumerate(identity_folders):
            self.identity_to_idx[identity_folder] = idx
        for identity_folder in identity_folders:
            identity_path = os.path.join(self.root_dir, identity_folder)
            original_images = [f for f in os.listdir(identity_path)
                               if f.lower().endswith(('.jpg', '.jpeg')) and
                               os.path.isfile(os.path.join(identity_path, f))]
            distorted_images = []
            distortion_path = os.path.join(identity_path, "distortion")
            if os.path.exists(distortion_path):
                distorted_images = [f for f in os.listdir(distortion_path)
                                   if f.lower().endswith(('.jpg', '.jpeg'))]
            all_images = [os.path.join(identity_path, img) for img in original_images]
            all_images += [os.path.join(distortion_path, img) for img in distorted_images]
            if len(all_images) >= 2:
                self.identity_data[identity_folder] = all_images
        print(f"TaskB {self.split}: Loaded {len(self.identity_data)} identities")

    def _generate_pairs(self):
        identities = list(self.identity_data.keys())
        if len(identities) < 2:
            return
        # Positive pairs
        for identity in identities:
            images = self.identity_data[identity]
            identity_idx = self.identity_to_idx[identity]
            pairs_generated = 0
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if pairs_generated < OptimizedArcFaceConfig.PAIRS_PER_IDENTITY:
                        self.pairs.append((images[i], images[j]))
                        self.labels.append(1)
                        self.identity_labels.append(identity_idx)
                        pairs_generated += 1
                    else:
                        break
                if pairs_generated >= OptimizedArcFaceConfig.PAIRS_PER_IDENTITY:
                    break
        # Negative pairs
        num_positive_pairs = self.labels.count(1)
        negative_pairs_generated = 0
        for i, identity1 in enumerate(identities):
            for j, identity2 in enumerate(identities[i + 1:], i + 1):
                if negative_pairs_generated >= num_positive_pairs:
                    break
                images1 = self.identity_data[identity1]
                images2 = self.identity_data[identity2]
                for img1 in images1:
                    for img2 in images2:
                        if negative_pairs_generated < num_positive_pairs:
                            self.pairs.append((img1, img2))
                            self.labels.append(0)
                            self.identity_labels.append(self.identity_to_idx[identity1])
                            negative_pairs_generated += 1
                        else:
                            break
                    if negative_pairs_generated >= num_positive_pairs:
                        break
            if negative_pairs_generated >= num_positive_pairs:
                break
        print(f"TaskB {self.split}: Generated {len(self.pairs)} pairs "
              f"(Positive: {self.labels.count(1)}, Negative: {self.labels.count(0)})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        identity_label = self.identity_labels[idx]
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            img1 = Image.new('RGB', (OptimizedArcFaceConfig.IMG_SIZE, OptimizedArcFaceConfig.IMG_SIZE), (0, 0, 0))
            img2 = Image.new('RGB', (OptimizedArcFaceConfig.IMG_SIZE, OptimizedArcFaceConfig.IMG_SIZE), (0, 0, 0))
        if self.transform:
            if OptimizedArcFaceConfig.USE_ALBUMENTATIONS:
                img1 = self.transform(image=np.array(img1))['image']
                img2 = self.transform(image=np.array(img2))['image']
            else:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        return img1, img2, label, identity_label

def get_research_optimized_transforms():
    """Get research-optimized transforms for both tasks."""
    if OptimizedArcFaceConfig.USE_ALBUMENTATIONS:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        train_transform = A.Compose([
            A.Resize(OptimizedArcFaceConfig.IMG_SIZE, OptimizedArcFaceConfig.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=OptimizedArcFaceConfig.ROTATION_LIMIT, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=OptimizedArcFaceConfig.PERSPECTIVE_PROB),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=OptimizedArcFaceConfig.ELASTIC_PROB),
            A.RandomBrightnessContrast(
                brightness_limit=OptimizedArcFaceConfig.BRIGHTNESS_LIMIT,
                contrast_limit=OptimizedArcFaceConfig.CONTRAST_LIMIT,
                p=0.6
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.CLAHE(p=0.3),
            A.Sharpen(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        val_transform = A.Compose([
            A.Resize(OptimizedArcFaceConfig.IMG_SIZE, OptimizedArcFaceConfig.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        import torchvision.transforms as transforms
        train_transform = transforms.Compose([
            transforms.Resize((OptimizedArcFaceConfig.IMG_SIZE, OptimizedArcFaceConfig.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(OptimizedArcFaceConfig.ROTATION_LIMIT),
            transforms.ColorJitter(
                brightness=OptimizedArcFaceConfig.BRIGHTNESS_LIMIT,
                contrast=OptimizedArcFaceConfig.CONTRAST_LIMIT,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=OptimizedArcFaceConfig.PERSPECTIVE_PROB),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((OptimizedArcFaceConfig.IMG_SIZE, OptimizedArcFaceConfig.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return train_transform, val_transform

def create_final_data_loaders():
    """Create research-optimized data loaders for final submission."""
    print("Creating final data loaders...")
    train_transform, val_transform = get_research_optimized_transforms()
    task_a_train_dataset = TaskADataset(
        root_dir=OptimizedArcFaceConfig.TASK_A_TRAIN_PATH,
        transform=train_transform,
        split='train'
    )
    task_a_val_dataset = TaskADataset(
        root_dir=OptimizedArcFaceConfig.TASK_A_VAL_PATH,
        transform=val_transform,
        split='val'
    )
    task_b_train_dataset = TaskBDataset(
        root_dir=OptimizedArcFaceConfig.TASK_B_TRAIN_PATH,
        transform=train_transform,
        split='train'
    )
    task_b_val_dataset = TaskBDataset(
        root_dir=OptimizedArcFaceConfig.TASK_B_VAL_PATH,
        transform=val_transform,
        split='val'
    )
    task_a_train_loader = DataLoader(
        task_a_train_dataset,
        batch_size=OptimizedArcFaceConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=OptimizedArcFaceConfig.NUM_WORKERS,
        pin_memory=OptimizedArcFaceConfig.PIN_MEMORY,
        drop_last=True
    )
    task_a_val_loader = DataLoader(
        task_a_val_dataset,
        batch_size=OptimizedArcFaceConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=OptimizedArcFaceConfig.NUM_WORKERS,
        pin_memory=OptimizedArcFaceConfig.PIN_MEMORY
    )
    task_b_train_loader = DataLoader(
        task_b_train_dataset,
        batch_size=OptimizedArcFaceConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=OptimizedArcFaceConfig.NUM_WORKERS,
        pin_memory=OptimizedArcFaceConfig.PIN_MEMORY,
        drop_last=True
    )
    task_b_val_loader = DataLoader(
        task_b_val_dataset,
        batch_size=OptimizedArcFaceConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=OptimizedArcFaceConfig.NUM_WORKERS,
        pin_memory=OptimizedArcFaceConfig.PIN_MEMORY
    )
    return {
        'task_a': {
            'train_loader': task_a_train_loader,
            'val_loader': task_a_val_loader,
            'train_dataset': task_a_train_dataset,
            'val_dataset': task_a_val_dataset
        },
        'task_b': {
            'train_loader': task_b_train_loader,
            'val_loader': task_b_val_loader,
            'train_dataset': task_b_train_dataset,
            'val_dataset': task_b_val_dataset
        }
    }

class DataLoaderManager:
    """Manager class for handling data loading operations."""
    def __init__(self, config):
        self.config = config
        self.data_loaders = None

    def initialize_data_loaders(self):
        try:
            print("Initializing data loaders...")
            self.data_loaders = create_final_data_loaders()
            print("Data loaders created successfully!")
            self._verify_data_loaders()
            return True
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            print("Please verify dataset paths and structure")
            return False

    def _verify_data_loaders(self):
        try:
            sample_batch = next(iter(self.data_loaders['task_a']['train_loader']))
            print(f"Task A batch shape: {sample_batch[0].shape}")
            sample_batch = next(iter(self.data_loaders['task_b']['train_loader']))
            print(f"Task B batch shapes: {sample_batch[0].shape}, {sample_batch[1].shape}")
            print("Data loader verification completed successfully")
        except Exception as e:
            print(f"Data loader verification failed: {e}")

    def get_data_loaders(self):
        if self.data_loaders is None:
            self.initialize_data_loaders()
        return self.data_loaders
