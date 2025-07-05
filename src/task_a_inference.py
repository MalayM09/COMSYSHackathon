import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.model_architecture import FinalHybridModel
from src.config import OptimizedArcFaceConfig as Config

# --- Dataset for Inference ---
class TaskAInferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label_name in ['male', 'female']:
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(label_dir, fname), 0 if label_name == 'male' else 1))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if Config.USE_ALBUMENTATIONS and self.transform:
            img = self.transform(image=np.array(img))['image']
        elif self.transform:
            img = self.transform(img)
        return img, label, img_path

def main(weights_path, data_dir, output_csv, use_joint=False):
    # Load model
    model = FinalHybridModel(Config)
    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE))
    model.eval()
    model.to(Config.DEVICE)

    # Get transforms
    from src.data_loading import get_research_optimized_transforms
    _, val_transform = get_research_optimized_transforms()

    # Dataset and loader
    dataset = TaskAInferenceDataset(data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # Inference
    results = []
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Task A Inference"):
            imgs = imgs.to(Config.DEVICE)
            outputs = model(imgs, task='gender')
            preds = torch.argmax(outputs['gender'], dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            for path, pred, label in zip(paths, preds, labels):
                results.append({'image': path, 'predicted_label': int(pred), 'true_label': int(label)})
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Task A predictions saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth file (best_gender.pth or best_joint.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Task_A/val')
    parser.add_argument('--output_csv', type=str, default='results_task_a.csv')
    args = parser.parse_args()
    main(args.weights, args.data_dir, args.output_csv)
