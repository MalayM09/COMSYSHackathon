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
class TaskBInferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, pairs_per_identity=10):
        self.pairs = []
        self.transform = transform
        identity_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        for identity in identity_folders:
            identity_path = os.path.join(root_dir, identity)
            originals = [os.path.join(identity_path, f) for f in os.listdir(identity_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(identity_path, f))]
            distortion_path = os.path.join(identity_path, 'distortion')
            distorteds = []
            if os.path.exists(distortion_path):
                distorteds = [os.path.join(distortion_path, f) for f in os.listdir(distortion_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            all_imgs = originals + distorteds
            # Positive pairs
            for i in range(min(len(all_imgs), pairs_per_identity)):
                for j in range(i+1, min(len(all_imgs), pairs_per_identity)):
                    self.pairs.append((all_imgs[i], all_imgs[j], 1))
            # Negative pairs (sampled)
        # For simplicity, only positive pairs here; you can add negative pairs similarly if needed

    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if Config.USE_ALBUMENTATIONS and self.transform:
            img1 = self.transform(image=np.array(img1))['image']
            img2 = self.transform(image=np.array(img2))['image']
        elif self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label, img1_path, img2_path

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
    dataset = TaskBInferenceDataset(data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # Inference
    results = []
    with torch.no_grad():
        for img1, img2, labels, path1, path2 in tqdm(loader, desc="Task B Inference"):
            img1 = img1.to(Config.DEVICE)
            img2 = img2.to(Config.DEVICE)
            emb1 = model(img1, task='embedding')['embedding']
            emb2 = model(img2, task='embedding')['embedding']
            similarity = torch.cosine_similarity(emb1, emb2, dim=1).cpu().numpy()
            preds = (similarity > 0.5).astype(int)  # You can use a better threshold
            labels = np.array(labels)
            for p1, p2, pred, label, sim in zip(path1, path2, preds, labels, similarity):
                results.append({'image1': p1, 'image2': p2, 'predicted_label': int(pred), 'true_label': int(label), 'similarity': float(sim)})
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Task B predictions saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth file (best_arcface.pth or best_joint.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Task_B/val')
    parser.add_argument('--output_csv', type=str, default='results_task_b.csv')
    args = parser.parse_args()
    main(args.weights, args.data_dir, args.output_csv)
