<<<<<<< HEAD
# ==============================================
# Sequential Fine-Tuning with Model Ensemble
# Author: Dharun | GPU: RTX 3050 (6GB)
# MULTI-MODEL ENSEMBLE FOR MAXIMUM ACCURACY
# ==============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 🚀 PATH CONFIGURATION - MODIFY THESE PATHS
# ==============================================
DATASET_PATHS = {
    'train': "combined_dataset/train",  # Change this to your train path
    'val': "combined_dataset/val",      # Change this to your validation path  
    'test': "combined_dataset/test"     # Change this to your test path
}

# Model save directory
MODEL_SAVE_DIR = "trained_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==============================================
# 🔄 Configuration - UPDATED WITH XCEPTION & EFFICIENTNET-B0
# ==============================================
MODEL_CONFIGS = [
    {
        'name': 'EfficientNet-B4',
        'model_name': 'efficientnet_b4',
        'input_size': 380,
        'batch_size': 4,
        'lr': 1e-4,
        'epochs': 8,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_efficientnet_b4.pth')
    },
    {
        'name': 'EfficientNet-B0',
        'model_name': 'efficientnet_b0',
        'input_size': 224,
        'batch_size': 16,
        'lr': 2e-4,
        'epochs': 6,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_efficientnet_b0.pth')
    },
    {
        'name': 'Xception',
        'model_name': 'xception',
        'input_size': 299,
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 7,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_xception.pth')
    },
    {
        'name': 'ResNet50',
        'model_name': 'resnet50',
        'input_size': 224,
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 6,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_resnet50.pth')
    },
    {
        'name': 'ConvNeXt-Tiny',
        'model_name': 'convnext_tiny',
        'input_size': 224,
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 6,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_convnext_tiny.pth')
    },
    {
        'name': 'MobileNetV3-Large',
        'model_name': 'mobilenetv3_large_100',
        'input_size': 224,
        'batch_size': 16,
        'lr': 2e-4,
        'epochs': 5,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_mobilenetv3_large.pth')
    }
]

# ==============================================
# 🔄 Dataset Class
# ==============================================
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, balance_data=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory {root_dir} does not exist!")
        
        real_count = 0
        fake_count = 0
        
        for label in ['real', 'fake']:
            folder = os.path.join(root_dir, label)
            if not os.path.exists(folder):
                print(f"⚠️ Warning: {folder} does not exist! Skipping...")
                continue
                
            for img_file in os.listdir(folder):
                if img_file.lower().endswith(('jpg', 'png', 'jpeg')):
                    self.samples.append(os.path.join(folder, img_file))
                    label_val = 0 if label == 'real' else 1
                    self.labels.append(label_val)
                    if label == 'real':
                        real_count += 1
                    else:
                        fake_count += 1
        
        print(f"📁 Loaded {len(self.samples)} images from {root_dir}")
        print(f"   Real: {real_count}, Fake: {fake_count}")
        
        if balance_data and real_count > 0 and fake_count > 0:
            total = real_count + fake_count
            self.class_weights = [total/real_count, total/fake_count]
            print(f"⚖️ Class weights - Real: {self.class_weights[0]:.2f}, Fake: {self.class_weights[1]:.2f}")
        else:
            self.class_weights = [1.0, 1.0]
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

# ==============================================
# 🔄 Enhanced Model Creation with Custom Heads
# ==============================================
def create_enhanced_model(model_name, num_classes=2):
    """Create enhanced model with custom head"""
    
    if 'xception' in model_name.lower():
        # Xception model
        model = timm.create_model('xception', pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    elif 'efficientnet_b4' in model_name.lower():
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    elif 'efficientnet_b0' in model_name.lower():
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    elif 'resnet' in model_name.lower():
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    elif 'convnext' in model_name.lower():
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    else:  # MobileNet and others
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model
    
    # Combine model and head
    class CustomModel(nn.Module):
        def __init__(self, backbone, head):
            super(CustomModel, self).__init__()
            self.backbone = backbone
            self.head = head
            
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
    
    return CustomModel(model, head)

# ==============================================
# 🔄 Focal Loss for Imbalanced Data
# ==============================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==============================================
# 🔄 Individual Model Training
# ==============================================
def train_single_model(model_config, model_index):
    """Train a single model from the sequence"""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING MODEL {model_index+1}/{len(MODEL_CONFIGS)}: {model_config['name']}")
    print(f"{'='*60}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get transforms for this model
    input_size = model_config['input_size']
    train_transform = transforms.Compose([
        transforms.Resize((input_size + 20, input_size + 20)),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets with configured paths
    try:
        train_dataset = DeepfakeDataset(DATASET_PATHS['train'], transform=train_transform, balance_data=True)
        val_dataset = DeepfakeDataset(DATASET_PATHS['val'], transform=val_transform, balance_data=False)
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return None

    # DataLoaders
    sample_weights = [train_dataset.class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config['batch_size'] * 2,
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Create model
    try:
        model = create_enhanced_model(model_config['model_name'])
        print(f"✅ {model_config['name']} loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading {model_config['name']}: {e}")
        return None

    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"📊 Trainable Parameters: {trainable_params:,}")

    # Loss and optimizer
    class_weights_tensor = torch.tensor([1.36, 1.0], device=device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    
    # Differential learning rates
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': model_config['lr'] / 10, 'weight_decay': 0.01},
        {'params': head_params, 'lr': model_config['lr'], 'weight_decay': 0.001}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config['epochs'])

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(model_config['epochs']):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Training
        model.train()
        running_loss, train_preds, train_labels = 0.0, [], []

        loop = tqdm(train_loader, desc=f"{model_config['name']} Epoch {epoch+1}/{model_config['epochs']} [Train]")
        for batch_idx, (imgs, labels) in enumerate(loop):
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            current_loss = running_loss / (batch_idx + 1)
            current_acc = accuracy_score(train_labels, train_preds)
            loop.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc*100:.2f}%'})

        # Validation
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        running_val_loss = 0.0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, [p[1] for p in val_probs])
        val_loss = running_val_loss / len(val_loader)

        # Update history
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f"📊 Epoch {epoch+1}: Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"🎉 New best {model_config['name']}! Val Acc: {val_acc*100:.2f}%")

        scheduler.step()

    # Save the trained model
    if best_model_state is not None:
        model_save_path = model_config['save_path']
        torch.save({
            'model_state_dict': best_model_state,
            'model_name': model_config['model_name'],
            'input_size': model_config['input_size'],
            'val_acc': best_val_acc,
            'val_auc': history['val_auc'][-1],
            'history': history
        }, model_save_path)
        print(f"💾 Saved {model_config['name']} as {model_save_path}")
        
        return {
            'name': model_config['name'],
            'model_path': model_save_path,
            'model_name': model_config['model_name'],
            'input_size': model_config['input_size'],
            'val_acc': best_val_acc,
            'val_auc': history['val_auc'][-1]
        }
    
    return None

# ==============================================
# 🔄 Model Ensemble Class
# ==============================================
class DeepfakeEnsemble:
    def __init__(self, model_configs):
        self.models = []
        self.model_configs = model_configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Load all trained models"""
        print(f"\n{'='*60}")
        print("🔄 LOADING ENSEMBLE MODELS")
        print(f"{'='*60}")
        
        for config in self.model_configs:
            try:
                # Load checkpoint
                checkpoint = torch.load(config['model_path'], map_location='cpu')
                
                # Create model architecture
                model = create_enhanced_model(config['model_name'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()
                
                self.models.append({
                    'model': model,
                    'name': config['name'],
                    'input_size': config['input_size'],
                    'weight': config['val_acc']  # Use validation accuracy as weight
                })
                
                print(f"✅ Loaded {config['name']} (Val Acc: {config['val_acc']*100:.2f}%)")
                
            except Exception as e:
                print(f"❌ Failed to load {config['name']}: {e}")
    
    def predict_ensemble(self, dataloader, method='weighted_vote'):
        """Make predictions using ensemble methods"""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                batch_probs = []
                
                # Get predictions from each model
                for model_info in self.models:
                    model = model_info['model']
                    input_size = model_info['input_size']
                    
                    # Resize if needed
                    if imgs.shape[2] != input_size:
                        resized_imgs = torch.nn.functional.interpolate(
                            imgs, size=(input_size, input_size), mode='bilinear', align_corners=False
                        )
                        outputs = model(resized_imgs)
                    else:
                        outputs = model(imgs)
                    
                    probs = torch.softmax(outputs, dim=1)
                    batch_probs.append(probs.cpu().numpy())
                
                # Convert to numpy arrays
                batch_probs = np.array(batch_probs)  # shape: (n_models, batch_size, n_classes)
                
                if method == 'weighted_vote':
                    # Weighted average based on validation accuracy
                    weights = np.array([m['weight'] for m in self.models])
                    weights = weights / weights.sum()
                    
                    # Calculate weighted average probabilities
                    weighted_probs = np.zeros_like(batch_probs[0])
                    for i, weight in enumerate(weights):
                        weighted_probs += batch_probs[i] * weight
                    
                    final_preds = np.argmax(weighted_probs, axis=1)
                    all_probabilities.extend(weighted_probs)
                    
                elif method == 'majority_vote':
                    # Majority voting
                    model_preds = np.argmax(batch_probs, axis=2)  # shape: (n_models, batch_size)
                    final_preds = []
                    for i in range(model_preds.shape[1]):
                        votes = model_preds[:, i]
                        final_preds.append(np.bincount(votes).argmax())
                    
                    # Use average probabilities for confidence
                    avg_probs = np.mean(batch_probs, axis=0)
                    all_probabilities.extend(avg_probs)
                
                elif method == 'max_prob':
                    # Use model with highest confidence for each sample
                    max_conf_preds = []
                    max_conf_probs = []
                    for i in range(batch_probs.shape[1]):
                        sample_probs = batch_probs[:, i, :]  # All models' probabilities for this sample
                        max_conf_idx = np.argmax(np.max(sample_probs, axis=1))
                        max_conf_preds.append(np.argmax(sample_probs[max_conf_idx]))
                        max_conf_probs.append(sample_probs[max_conf_idx])
                    
                    final_preds = max_conf_preds
                    all_probabilities.extend(max_conf_probs)
                
                all_predictions.extend(final_preds)
                all_labels.extend(labels.numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)
    
    def evaluate_ensemble(self, dataloader, method='weighted_vote'):
        """Comprehensive ensemble evaluation"""
        print(f"\n{'='*60}")
        print(f"🎯 ENSEMBLE EVALUATION ({method.upper()})")
        print(f"{'='*60}")
        
        predictions, probabilities, true_labels = self.predict_ensemble(dataloader, method)
        
        accuracy = accuracy_score(true_labels, predictions)
        auc_score = roc_auc_score(true_labels, probabilities[:, 1])
        
        print(f"📊 Ensemble Accuracy: {accuracy*100:.2f}%")
        print(f"📊 Ensemble AUC: {auc_score:.4f}")
        print(f"📊 Ensemble Method: {method}")
        print("\n📋 Classification Report:")
        print(classification_report(true_labels, predictions, target_names=['Real', 'Fake'], digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {method.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return accuracy, auc_score, predictions, probabilities, true_labels

# ==============================================
# 🔄 Visualization Functions
# ==============================================
def plot_training_history(histories):
    """Plot training history for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (model_name, history) in enumerate(histories.items()):
        # Accuracy
        axes[0].plot(history['train_acc'], label=f'{model_name} Train', alpha=0.7)
        axes[0].plot(history['val_acc'], label=f'{model_name} Val', linestyle='--', alpha=0.7)
        axes[0].set_title('Accuracy')
        axes[0].legend()
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        
        # Loss
        axes[1].plot(history['train_loss'], label=f'{model_name} Train', alpha=0.7)
        axes[1].plot(history['val_loss'], label=f'{model_name} Val', linestyle='--', alpha=0.7)
        axes[1].set_title('Loss')
        axes[1].legend()
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        
        # AUC
        axes[2].plot(history['val_auc'], label=model_name, alpha=0.7)
        axes[2].set_title('Validation AUC')
        axes[2].legend()
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
    
    plt.tight_layout()
    plt.show()

def compare_model_performance(trained_models):
    """Compare performance of all trained models"""
    model_names = [model['name'] for model in trained_models]
    val_accs = [model['val_acc'] * 100 for model in trained_models]
    val_aucs = [model['val_auc'] * 100 for model in trained_models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, val_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax1.set_title('Model Validation Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # AUC comparison
    bars2 = ax2.bar(model_names, val_aucs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('Model Validation AUC Comparison')
    ax2.set_ylabel('AUC (%)')
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# ==============================================
# 🔄 Main Training Pipeline
# ==============================================
def sequential_fine_tuning():
    """Main pipeline for sequential fine-tuning and ensemble"""
    print(f"{'='*80}")
    print("🎯 DEEPFAKE DETECTION - SEQUENTIAL FINE-TUNING WITH ENSEMBLE")
    print(f"{'='*80}")
    
    # Check dataset paths
    print("🔍 Checking dataset paths...")
    for split, path in DATASET_PATHS.items():
        if os.path.exists(path):
            print(f"   ✅ {split}: {path}")
        else:
            print(f"   ❌ {split}: {path} - PATH NOT FOUND!")
            return
    
    trained_models = []
    training_histories = {}
    
    # Step 1: Sequential Fine-Tuning
    for i, config in enumerate(MODEL_CONFIGS):
        result = train_single_model(config, i)
        if result is not None:
            trained_models.append(result)
            # Load history for plotting
            checkpoint = torch.load(config['save_path'])
            training_histories[config['name']] = checkpoint['history']
        
        # Clear memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not trained_models:
        print("❌ No models were successfully trained!")
        return
    
    # Step 2: Plot training results
    print(f"\n{'='*80}")
    print("📊 VISUALIZING TRAINING RESULTS")
    print(f"{'='*80}")
    
    plot_training_history(training_histories)
    compare_model_performance(trained_models)
    
    # Step 3: Create Ensemble
    print(f"\n{'='*80}")
    print("🤝 CREATING MODEL ENSEMBLE")
    print(f"{'='*80}")
    
    ensemble = DeepfakeEnsemble(trained_models)
    ensemble.load_models()
    
    # Step 4: Evaluate Ensemble
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        val_dataset = DeepfakeDataset(DATASET_PATHS['val'], transform=val_transform, balance_data=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Test different ensemble methods
        methods = ['weighted_vote', 'majority_vote', 'max_prob']
        best_method = None
        best_accuracy = 0
        ensemble_results = {}
        
        for method in methods:
            accuracy, auc, preds, probs, labels = ensemble.evaluate_ensemble(val_loader, method)
            ensemble_results[method] = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': preds,
                'probabilities': probs,
                'true_labels': labels
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
        
        print(f"\n🎉 BEST ENSEMBLE METHOD: {best_method.upper()}")
        print(f"🏆 Best Ensemble Accuracy: {best_accuracy*100:.2f}%")
        
        # Save ensemble results
        ensemble_save_path = os.path.join(MODEL_SAVE_DIR, 'ensemble_results.npy')
        np.save(ensemble_save_path, ensemble_results)
        print(f"💾 Ensemble results saved to: {ensemble_save_path}")
        
    except Exception as e:
        print(f"❌ Error during ensemble evaluation: {e}")

# ==============================================
# 🚀 RUN THE TRAINING
# ==============================================
if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🎯 CUDA is available! Using GPU: {torch.cuda.get_device_name()}")
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (training will be slow!)")
    
    # Run the training pipeline
=======
# ==============================================
# Sequential Fine-Tuning with Model Ensemble
# Author: Dharun | GPU: RTX 3050 (6GB)
# MULTI-MODEL ENSEMBLE FOR MAXIMUM ACCURACY
# ==============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 🚀 PATH CONFIGURATION - MODIFY THESE PATHS
# ==============================================
DATASET_PATHS = {
    'train': "combined_dataset/train",  # Change this to your train path
    'val': "combined_dataset/val",      # Change this to your validation path  
    'test': "combined_dataset/test"     # Change this to your test path
}

# Model save directory
MODEL_SAVE_DIR = "trained_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==============================================
# 🔄 Configuration - UPDATED WITH XCEPTION & EFFICIENTNET-B0
# ==============================================
MODEL_CONFIGS = [
    {
        'name': 'EfficientNet-B4',
        'model_name': 'efficientnet_b4',
        'input_size': 380,
        'batch_size': 4,
        'lr': 1e-4,
        'epochs': 8,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_efficientnet_b4.pth')
    },
    {
        'name': 'EfficientNet-B0',
        'model_name': 'efficientnet_b0',
        'input_size': 224,
        'batch_size': 16,
        'lr': 2e-4,
        'epochs': 6,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_efficientnet_b0.pth')
    },
    {
        'name': 'Xception',
        'model_name': 'xception',
        'input_size': 299,
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 7,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_xception.pth')
    },
    {
        'name': 'ResNet50',
        'model_name': 'resnet50',
        'input_size': 224,
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 6,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_resnet50.pth')
    },
    {
        'name': 'ConvNeXt-Tiny',
        'model_name': 'convnext_tiny',
        'input_size': 224,
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 6,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_convnext_tiny.pth')
    },
    {
        'name': 'MobileNetV3-Large',
        'model_name': 'mobilenetv3_large_100',
        'input_size': 224,
        'batch_size': 16,
        'lr': 2e-4,
        'epochs': 5,
        'save_path': os.path.join(MODEL_SAVE_DIR, 'best_mobilenetv3_large.pth')
    }
]

# ==============================================
# 🔄 Dataset Class
# ==============================================
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, balance_data=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory {root_dir} does not exist!")
        
        real_count = 0
        fake_count = 0
        
        for label in ['real', 'fake']:
            folder = os.path.join(root_dir, label)
            if not os.path.exists(folder):
                print(f"⚠️ Warning: {folder} does not exist! Skipping...")
                continue
                
            for img_file in os.listdir(folder):
                if img_file.lower().endswith(('jpg', 'png', 'jpeg')):
                    self.samples.append(os.path.join(folder, img_file))
                    label_val = 0 if label == 'real' else 1
                    self.labels.append(label_val)
                    if label == 'real':
                        real_count += 1
                    else:
                        fake_count += 1
        
        print(f"📁 Loaded {len(self.samples)} images from {root_dir}")
        print(f"   Real: {real_count}, Fake: {fake_count}")
        
        if balance_data and real_count > 0 and fake_count > 0:
            total = real_count + fake_count
            self.class_weights = [total/real_count, total/fake_count]
            print(f"⚖️ Class weights - Real: {self.class_weights[0]:.2f}, Fake: {self.class_weights[1]:.2f}")
        else:
            self.class_weights = [1.0, 1.0]
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

# ==============================================
# 🔄 Enhanced Model Creation with Custom Heads
# ==============================================
def create_enhanced_model(model_name, num_classes=2):
    """Create enhanced model with custom head"""
    
    if 'xception' in model_name.lower():
        # Xception model
        model = timm.create_model('xception', pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    elif 'efficientnet_b4' in model_name.lower():
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    elif 'efficientnet_b0' in model_name.lower():
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    elif 'resnet' in model_name.lower():
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    elif 'convnext' in model_name.lower():
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = model.num_features
        head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    else:  # MobileNet and others
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model
    
    # Combine model and head
    class CustomModel(nn.Module):
        def __init__(self, backbone, head):
            super(CustomModel, self).__init__()
            self.backbone = backbone
            self.head = head
            
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
    
    return CustomModel(model, head)

# ==============================================
# 🔄 Focal Loss for Imbalanced Data
# ==============================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==============================================
# 🔄 Individual Model Training
# ==============================================
def train_single_model(model_config, model_index):
    """Train a single model from the sequence"""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING MODEL {model_index+1}/{len(MODEL_CONFIGS)}: {model_config['name']}")
    print(f"{'='*60}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get transforms for this model
    input_size = model_config['input_size']
    train_transform = transforms.Compose([
        transforms.Resize((input_size + 20, input_size + 20)),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets with configured paths
    try:
        train_dataset = DeepfakeDataset(DATASET_PATHS['train'], transform=train_transform, balance_data=True)
        val_dataset = DeepfakeDataset(DATASET_PATHS['val'], transform=val_transform, balance_data=False)
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return None

    # DataLoaders
    sample_weights = [train_dataset.class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config['batch_size'] * 2,
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Create model
    try:
        model = create_enhanced_model(model_config['model_name'])
        print(f"✅ {model_config['name']} loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading {model_config['name']}: {e}")
        return None

    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"📊 Trainable Parameters: {trainable_params:,}")

    # Loss and optimizer
    class_weights_tensor = torch.tensor([1.36, 1.0], device=device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    
    # Differential learning rates
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': model_config['lr'] / 10, 'weight_decay': 0.01},
        {'params': head_params, 'lr': model_config['lr'], 'weight_decay': 0.001}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config['epochs'])

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(model_config['epochs']):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Training
        model.train()
        running_loss, train_preds, train_labels = 0.0, [], []

        loop = tqdm(train_loader, desc=f"{model_config['name']} Epoch {epoch+1}/{model_config['epochs']} [Train]")
        for batch_idx, (imgs, labels) in enumerate(loop):
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            current_loss = running_loss / (batch_idx + 1)
            current_acc = accuracy_score(train_labels, train_preds)
            loop.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc*100:.2f}%'})

        # Validation
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        running_val_loss = 0.0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, [p[1] for p in val_probs])
        val_loss = running_val_loss / len(val_loader)

        # Update history
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f"📊 Epoch {epoch+1}: Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"🎉 New best {model_config['name']}! Val Acc: {val_acc*100:.2f}%")

        scheduler.step()

    # Save the trained model
    if best_model_state is not None:
        model_save_path = model_config['save_path']
        torch.save({
            'model_state_dict': best_model_state,
            'model_name': model_config['model_name'],
            'input_size': model_config['input_size'],
            'val_acc': best_val_acc,
            'val_auc': history['val_auc'][-1],
            'history': history
        }, model_save_path)
        print(f"💾 Saved {model_config['name']} as {model_save_path}")
        
        return {
            'name': model_config['name'],
            'model_path': model_save_path,
            'model_name': model_config['model_name'],
            'input_size': model_config['input_size'],
            'val_acc': best_val_acc,
            'val_auc': history['val_auc'][-1]
        }
    
    return None

# ==============================================
# 🔄 Model Ensemble Class
# ==============================================
class DeepfakeEnsemble:
    def __init__(self, model_configs):
        self.models = []
        self.model_configs = model_configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Load all trained models"""
        print(f"\n{'='*60}")
        print("🔄 LOADING ENSEMBLE MODELS")
        print(f"{'='*60}")
        
        for config in self.model_configs:
            try:
                # Load checkpoint
                checkpoint = torch.load(config['model_path'], map_location='cpu')
                
                # Create model architecture
                model = create_enhanced_model(config['model_name'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()
                
                self.models.append({
                    'model': model,
                    'name': config['name'],
                    'input_size': config['input_size'],
                    'weight': config['val_acc']  # Use validation accuracy as weight
                })
                
                print(f"✅ Loaded {config['name']} (Val Acc: {config['val_acc']*100:.2f}%)")
                
            except Exception as e:
                print(f"❌ Failed to load {config['name']}: {e}")
    
    def predict_ensemble(self, dataloader, method='weighted_vote'):
        """Make predictions using ensemble methods"""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                batch_probs = []
                
                # Get predictions from each model
                for model_info in self.models:
                    model = model_info['model']
                    input_size = model_info['input_size']
                    
                    # Resize if needed
                    if imgs.shape[2] != input_size:
                        resized_imgs = torch.nn.functional.interpolate(
                            imgs, size=(input_size, input_size), mode='bilinear', align_corners=False
                        )
                        outputs = model(resized_imgs)
                    else:
                        outputs = model(imgs)
                    
                    probs = torch.softmax(outputs, dim=1)
                    batch_probs.append(probs.cpu().numpy())
                
                # Convert to numpy arrays
                batch_probs = np.array(batch_probs)  # shape: (n_models, batch_size, n_classes)
                
                if method == 'weighted_vote':
                    # Weighted average based on validation accuracy
                    weights = np.array([m['weight'] for m in self.models])
                    weights = weights / weights.sum()
                    
                    # Calculate weighted average probabilities
                    weighted_probs = np.zeros_like(batch_probs[0])
                    for i, weight in enumerate(weights):
                        weighted_probs += batch_probs[i] * weight
                    
                    final_preds = np.argmax(weighted_probs, axis=1)
                    all_probabilities.extend(weighted_probs)
                    
                elif method == 'majority_vote':
                    # Majority voting
                    model_preds = np.argmax(batch_probs, axis=2)  # shape: (n_models, batch_size)
                    final_preds = []
                    for i in range(model_preds.shape[1]):
                        votes = model_preds[:, i]
                        final_preds.append(np.bincount(votes).argmax())
                    
                    # Use average probabilities for confidence
                    avg_probs = np.mean(batch_probs, axis=0)
                    all_probabilities.extend(avg_probs)
                
                elif method == 'max_prob':
                    # Use model with highest confidence for each sample
                    max_conf_preds = []
                    max_conf_probs = []
                    for i in range(batch_probs.shape[1]):
                        sample_probs = batch_probs[:, i, :]  # All models' probabilities for this sample
                        max_conf_idx = np.argmax(np.max(sample_probs, axis=1))
                        max_conf_preds.append(np.argmax(sample_probs[max_conf_idx]))
                        max_conf_probs.append(sample_probs[max_conf_idx])
                    
                    final_preds = max_conf_preds
                    all_probabilities.extend(max_conf_probs)
                
                all_predictions.extend(final_preds)
                all_labels.extend(labels.numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)
    
    def evaluate_ensemble(self, dataloader, method='weighted_vote'):
        """Comprehensive ensemble evaluation"""
        print(f"\n{'='*60}")
        print(f"🎯 ENSEMBLE EVALUATION ({method.upper()})")
        print(f"{'='*60}")
        
        predictions, probabilities, true_labels = self.predict_ensemble(dataloader, method)
        
        accuracy = accuracy_score(true_labels, predictions)
        auc_score = roc_auc_score(true_labels, probabilities[:, 1])
        
        print(f"📊 Ensemble Accuracy: {accuracy*100:.2f}%")
        print(f"📊 Ensemble AUC: {auc_score:.4f}")
        print(f"📊 Ensemble Method: {method}")
        print("\n📋 Classification Report:")
        print(classification_report(true_labels, predictions, target_names=['Real', 'Fake'], digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {method.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return accuracy, auc_score, predictions, probabilities, true_labels

# ==============================================
# 🔄 Visualization Functions
# ==============================================
def plot_training_history(histories):
    """Plot training history for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (model_name, history) in enumerate(histories.items()):
        # Accuracy
        axes[0].plot(history['train_acc'], label=f'{model_name} Train', alpha=0.7)
        axes[0].plot(history['val_acc'], label=f'{model_name} Val', linestyle='--', alpha=0.7)
        axes[0].set_title('Accuracy')
        axes[0].legend()
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        
        # Loss
        axes[1].plot(history['train_loss'], label=f'{model_name} Train', alpha=0.7)
        axes[1].plot(history['val_loss'], label=f'{model_name} Val', linestyle='--', alpha=0.7)
        axes[1].set_title('Loss')
        axes[1].legend()
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        
        # AUC
        axes[2].plot(history['val_auc'], label=model_name, alpha=0.7)
        axes[2].set_title('Validation AUC')
        axes[2].legend()
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
    
    plt.tight_layout()
    plt.show()

def compare_model_performance(trained_models):
    """Compare performance of all trained models"""
    model_names = [model['name'] for model in trained_models]
    val_accs = [model['val_acc'] * 100 for model in trained_models]
    val_aucs = [model['val_auc'] * 100 for model in trained_models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, val_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax1.set_title('Model Validation Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # AUC comparison
    bars2 = ax2.bar(model_names, val_aucs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('Model Validation AUC Comparison')
    ax2.set_ylabel('AUC (%)')
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# ==============================================
# 🔄 Main Training Pipeline
# ==============================================
def sequential_fine_tuning():
    """Main pipeline for sequential fine-tuning and ensemble"""
    print(f"{'='*80}")
    print("🎯 DEEPFAKE DETECTION - SEQUENTIAL FINE-TUNING WITH ENSEMBLE")
    print(f"{'='*80}")
    
    # Check dataset paths
    print("🔍 Checking dataset paths...")
    for split, path in DATASET_PATHS.items():
        if os.path.exists(path):
            print(f"   ✅ {split}: {path}")
        else:
            print(f"   ❌ {split}: {path} - PATH NOT FOUND!")
            return
    
    trained_models = []
    training_histories = {}
    
    # Step 1: Sequential Fine-Tuning
    for i, config in enumerate(MODEL_CONFIGS):
        result = train_single_model(config, i)
        if result is not None:
            trained_models.append(result)
            # Load history for plotting
            checkpoint = torch.load(config['save_path'])
            training_histories[config['name']] = checkpoint['history']
        
        # Clear memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not trained_models:
        print("❌ No models were successfully trained!")
        return
    
    # Step 2: Plot training results
    print(f"\n{'='*80}")
    print("📊 VISUALIZING TRAINING RESULTS")
    print(f"{'='*80}")
    
    plot_training_history(training_histories)
    compare_model_performance(trained_models)
    
    # Step 3: Create Ensemble
    print(f"\n{'='*80}")
    print("🤝 CREATING MODEL ENSEMBLE")
    print(f"{'='*80}")
    
    ensemble = DeepfakeEnsemble(trained_models)
    ensemble.load_models()
    
    # Step 4: Evaluate Ensemble
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        val_dataset = DeepfakeDataset(DATASET_PATHS['val'], transform=val_transform, balance_data=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Test different ensemble methods
        methods = ['weighted_vote', 'majority_vote', 'max_prob']
        best_method = None
        best_accuracy = 0
        ensemble_results = {}
        
        for method in methods:
            accuracy, auc, preds, probs, labels = ensemble.evaluate_ensemble(val_loader, method)
            ensemble_results[method] = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': preds,
                'probabilities': probs,
                'true_labels': labels
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
        
        print(f"\n🎉 BEST ENSEMBLE METHOD: {best_method.upper()}")
        print(f"🏆 Best Ensemble Accuracy: {best_accuracy*100:.2f}%")
        
        # Save ensemble results
        ensemble_save_path = os.path.join(MODEL_SAVE_DIR, 'ensemble_results.npy')
        np.save(ensemble_save_path, ensemble_results)
        print(f"💾 Ensemble results saved to: {ensemble_save_path}")
        
    except Exception as e:
        print(f"❌ Error during ensemble evaluation: {e}")

# ==============================================
# 🚀 RUN THE TRAINING
# ==============================================
if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🎯 CUDA is available! Using GPU: {torch.cuda.get_device_name()}")
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (training will be slow!)")
    
    # Run the training pipeline
>>>>>>> d6265bf9ccf1d66490ca26d8658512566d956537
    sequential_fine_tuning()