import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone for feature extraction."""
    def __init__(self, model_name='convnext_tiny', pretrained=True, num_features=768):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.num_features = num_features

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)
        flat = self.flatten(pooled)
        return flat

class ProvenGenderHead(nn.Module):
    """Gender classification head: MLP + BN + ReLU + Dropout."""
    def __init__(self, input_features, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class ArcFaceVerificationHead(nn.Module):
    """ArcFace embedding head: MLP + BN + ReLU + Dropout + L2 norm."""
    def __init__(self, input_features, embedding_dim=512, dropout_rate=0.4):
        super().__init__()
        self.embedding_network = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        emb = self.embedding_network(x)
        return F.normalize(emb, p=2, dim=1)

class ArcFaceLoss(nn.Module):
    """ArcFace loss: Additive angular margin loss for verification."""
    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine, device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output

class FinalHybridModel(nn.Module):
    """
    Final model: ConvNeXt backbone + Gender head + ArcFace head.
    Use task='gender', 'embedding', or 'both' in forward().
    """
    def __init__(self, config):
        super().__init__()
        self.backbone = ConvNeXtBackbone(
            model_name=config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_features=768
        )
        self.gender_head = ProvenGenderHead(
            input_features=768,
            num_classes=config.GENDER_CLASSES,
            dropout_rate=config.GENDER_DROPOUT
        )
        self.verification_head = ArcFaceVerificationHead(
            input_features=768,
            embedding_dim=config.EMBEDDING_DIM,
            dropout_rate=config.VERIFICATION_DROPOUT
        )
        self.arcface_loss = ArcFaceLoss(
            embedding_dim=config.EMBEDDING_DIM,
            num_classes=config.NUM_IDENTITIES,
            margin=config.ARCFACE_MARGIN,
            scale=config.ARCFACE_SCALE
        )

    def forward(self, x, task='both'):
        shared_features = self.backbone(x)
        outputs = {}
        if task in ['gender', 'both']:
            outputs['gender'] = self.gender_head(shared_features)
        if task in ['embedding', 'both']:
            outputs['embedding'] = self.verification_head(shared_features)
        return outputs

    def arcface_forward(self, x, labels):
        emb = self.forward(x, task='embedding')['embedding']
        arcface_logits = self.arcface_loss(emb, labels)
        return arcface_logits, emb

class FinalCombinedLoss(nn.Module):
    """
    Combined loss for joint training.
    Accepts a dict of outputs and a dict of targets.
    """
    def __init__(self, config):
        super().__init__()
        self.gender_loss = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        self.arcface_loss = nn.CrossEntropyLoss()
        self.verification_loss = nn.BCEWithLogitsLoss()
        self.gender_weight = config.GENDER_LOSS_WEIGHT
        self.arcface_weight = config.ARCFACE_LOSS_WEIGHT
        self.verification_weight = config.VERIFICATION_LOSS_WEIGHT

    def forward(self, outputs, targets):
        total_loss = 0.0
        loss_dict = {}
        if 'gender' in outputs and 'gender' in targets:
            gender_loss = self.gender_loss(outputs['gender'], targets['gender'])
            total_loss += self.gender_weight * gender_loss
            loss_dict['gender_loss'] = gender_loss.item()
        if 'arcface_logits' in outputs and 'identity' in targets:
            arcface_loss = self.arcface_loss(outputs['arcface_logits'], targets['identity'])
            total_loss += self.arcface_weight * arcface_loss
            loss_dict['arcface_loss'] = arcface_loss.item()
        if 'similarity' in outputs and 'verification' in targets:
            verification_loss = self.verification_loss(outputs['similarity'], targets['verification'].float())
            total_loss += self.verification_weight * verification_loss
            loss_dict['verification_loss'] = verification_loss.item()
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

def create_final_model(config):
    """
    Create and initialize the final hybrid model.
    """
    print("Creating final hybrid model...")
    model = FinalHybridModel(config)
    model = model.to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Final model created!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("Integrates gender and verification techniques")
    return model
