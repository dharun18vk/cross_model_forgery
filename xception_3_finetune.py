import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import timm

# Suppress warnings
warnings.filterwarnings('ignore')

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, balance_dataset=True):
        """
        Dataset for real and fake face images
        Args:
            real_dir: Directory containing real face images
            fake_dir: Directory containing fake face images
            transform: Image transformations
            balance_dataset: Whether to balance real and fake samples
        """
        self.real_images = []
        self.fake_images = []
        self.labels = []
        self.transform = transform
        
        # Load real images
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.real_images.append(os.path.join(real_dir, img_file))
        else:
            print(f"⚠️ Warning: Real directory not found: {real_dir}")
        
        # Load fake images
        if os.path.exists(fake_dir):
            for img_file in os.listdir(fake_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.fake_images.append(os.path.join(fake_dir, img_file))
        else:
            print(f"⚠️ Warning: Fake directory not found: {fake_dir}")
        
        # Balance dataset if requested
        if balance_dataset and self.real_images and self.fake_images:
            min_count = min(len(self.real_images), len(self.fake_images))
            if len(self.real_images) > min_count:
                self.real_images = np.random.choice(self.real_images, min_count, replace=False).tolist()
            if len(self.fake_images) > min_count:
                self.fake_images = np.random.choice(self.fake_images, min_count, replace=False).tolist()
        
        # Combine images and create labels
        self.all_images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)  # 0=real, 1=fake
        
        if len(self.all_images) == 0:
            print(f"❌ No images found in {real_dir} or {fake_dir}")
        else:
            print(f"✅ Loaded {len(self.real_images)} real images and {len(self.fake_images)} fake images")
            print(f"📊 Total: {len(self.all_images)} images")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return a placeholder image
            placeholder = torch.zeros(3, 299, 299)
            return placeholder, torch.tensor(label, dtype=torch.long)

class MultiDatasetDeepfakeDataset(Dataset):
    def __init__(self, dataset_dirs, transform=None, balance_dataset=True):
        """
        Dataset that combines multiple datasets
        Args:
            dataset_dirs: List of tuples (dataset_name, real_dir, fake_dir)
            transform: Image transformations
            balance_dataset: Whether to balance real and fake samples
        """
        self.all_images = []
        self.labels = []
        self.dataset_labels = []  # Track which dataset each image comes from
        self.transform = transform
        
        for dataset_name, real_dir, fake_dir in dataset_dirs:
            print(f"📁 Loading dataset: {dataset_name}")
            
            real_images = []
            fake_images = []
            
            # Load real images
            if os.path.exists(real_dir):
                for img_file in os.listdir(real_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        real_images.append(os.path.join(real_dir, img_file))
            else:
                print(f"⚠️ Warning: Real directory not found: {real_dir}")
            
            # Load fake images
            if os.path.exists(fake_dir):
                for img_file in os.listdir(fake_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        fake_images.append(os.path.join(fake_dir, img_file))
            else:
                print(f"⚠️ Warning: Fake directory not found: {fake_dir}")
            
            # Balance within dataset if requested
            if balance_dataset and real_images and fake_images:
                min_count = min(len(real_images), len(fake_images))
                if len(real_images) > min_count:
                    real_images = np.random.choice(real_images, min_count, replace=False).tolist()
                if len(fake_images) > min_count:
                    fake_images = np.random.choice(fake_images, min_count, replace=False).tolist()
            
            # Add to combined dataset
            self.all_images.extend(real_images)
            self.all_images.extend(fake_images)
            self.labels.extend([0] * len(real_images) + [1] * len(fake_images))
            self.dataset_labels.extend([dataset_name] * (len(real_images) + len(fake_images)))
            
            print(f"   ✅ {len(real_images)} real, {len(fake_images)} fake images")
        
        print(f"📊 Combined dataset: {len(self.all_images)} total images")
        print(f"📊 Dataset distribution: {pd.Series(self.dataset_labels).value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return a placeholder image
            placeholder = torch.zeros(3, 299, 299)
            return placeholder, torch.tensor(label, dtype=torch.long)

class DeepfakeModel(nn.Module):
    def __init__(self, model_name='xception', num_classes=2, pretrained=True):
        super(DeepfakeModel, self).__init__()
        
        self.model_name = model_name
        
        if model_name == 'xception':
            self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'convnext_small':
            self.backbone = models.convnext_small(pretrained=pretrained)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier
        if model_name == 'xception':
            # Larger classifier for Xception
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1024),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes)
            )
        else:
            # Standard classifier for other models
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes)
            )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class ProgressiveFineTuneTrainer:
    def __init__(self, model_name='xception', num_classes=2, device=None, pretrained_path=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = DeepfakeModel(model_name, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
        # Load pre-trained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"🔄 Loading pre-trained model from: {pretrained_path}")
            self.load_model(pretrained_path)
        else:
            print("🆕 Training from scratch (no pre-trained weights found)")
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': []
        }
    
    def load_model(self, checkpoint_path):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint
            
            # Load model state dict
            self.model.load_state_dict(model_state_dict, strict=False)
            
            print(f"✅ Successfully loaded pre-trained model")
            if 'val_f1' in checkpoint:
                print(f"📊 Previous best F1: {checkpoint['val_f1']:.4f}")
            if 'timestamp' in checkpoint:
                print(f"🕒 Previous training time: {checkpoint['timestamp']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🆕 Continuing with randomly initialized weights")
    
    def get_transforms(self, augment=True):
        """Define data transformations for training and validation"""
        # Xception expects 299x299 input
        input_size = 299
        resize_size = 331
        
        if augment:
            # Training transforms with moderate augmentation for fine-tuning
            transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(5),  # Reduced rotation for fine-tuning
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Reduced jitter
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Reduced affine
                transforms.GaussianBlur(3, sigma=(0.1, 1.0)),  # Reduced blur
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/Test transforms without augmentation
            transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def setup_optimizer(self, learning_rate=5e-6, weight_decay=1e-5, fine_tune_strategy='progressive'):
        """Setup optimizer with progressive fine-tuning strategy"""
        
        # Different strategies for progressive fine-tuning
        if fine_tune_strategy == 'progressive':
            # Stage 3: Very fine-grained tuning
            layer_groups = {
                'backbone_early': [],      # Early layers - almost frozen
                'backbone_mid': [],        # Middle layers - very low LR
                'backbone_late': [],       # Late layers - low LR
                'classifier': []           # Classifier - medium LR
            }
            
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    layer_groups['classifier'].append(param)
                elif 'backbone.conv4' in name or 'backbone.conv3' in name:
                    layer_groups['backbone_late'].append(param)
                elif 'backbone.conv2' in name or 'backbone.conv1' in name:
                    layer_groups['backbone_mid'].append(param)
                else:
                    layer_groups['backbone_early'].append(param)
            
            print(f"📊 Progressive Fine-tuning Parameter Groups:")
            print(f"   - Backbone Early (frozen): {len(layer_groups['backbone_early'])} parameters")
            print(f"   - Backbone Mid (very low LR): {len(layer_groups['backbone_mid'])} parameters")
            print(f"   - Backbone Late (low LR): {len(layer_groups['backbone_late'])} parameters")
            print(f"   - Classifier (medium LR): {len(layer_groups['classifier'])} parameters")
            
            self.optimizer = optim.AdamW([
                {'params': layer_groups['backbone_early'], 'lr': learning_rate * 0.001},  # Almost frozen
                {'params': layer_groups['backbone_mid'], 'lr': learning_rate * 0.01},     # Very low LR
                {'params': layer_groups['backbone_late'], 'lr': learning_rate * 0.1},     # Low LR
                {'params': layer_groups['classifier'], 'lr': learning_rate}               # Medium LR
            ], weight_decay=weight_decay)
            
        elif fine_tune_strategy == 'aggressive':
            # More aggressive fine-tuning for challenging datasets
            backbone_params = []
            classifier_params = []
            
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
            
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': learning_rate * 0.1},
                {'params': classifier_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
        
        # Gentle learning rate scheduler for fine-tuning
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.001
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            # Gentle gradient clipping for fine-tuning
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, accuracy, f1
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return epoch_loss, accuracy, precision, recall, f1, all_probabilities, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=20, save_dir='progressive_fine_tuned_model', patience=8):
        """Progressive fine-tuning loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        patience_counter = 0
        
        print(f"🚀 Starting progressive fine-tuning for {epochs} epochs on {self.device}")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        print(f"🤖 Model: {self.model_name}")
        print(f"🎯 Strategy: Progressive fine-tuning with gentle learning rates")
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, val_precision, val_recall, val_f1, val_probs, val_preds, val_labels = self.validate_epoch(val_loader)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            self.history['learning_rates'].append(current_lr)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Print epoch results
            print(f"✅ Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"🎯 Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"📈 Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            print(f"📉 Learning Rate: {current_lr:.2e}")
            
            # Save best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model(os.path.join(save_dir, '3rd_tuned_xception_model.pth'), epoch, val_f1)
                print(f"💾 New best model saved with F1: {val_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'), epoch, val_f1)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_model(os.path.join(save_dir, 'final_progressive_model.pth'), epoch, val_f1)
        print(f"\n🎉 Progressive fine-tuning completed! Best validation F1: {best_val_f1:.4f}")
        
        return self.history
    
    def save_model(self, path, epoch, val_f1):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'history': self.history,
            'model_name': self.model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'training_stage': 'progressive_fine_tune_4'
        }
        torch.save(checkpoint, path)
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Progressive Fine-tuning Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_title('Progressive Fine-tuning Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1', linewidth=2)
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', linewidth=2)
        axes[1, 0].set_title('Progressive Fine-tuning F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['learning_rates'], label='Learning Rate', linewidth=2, color='purple')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_loader, save_path=None):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Class-wise metrics
        class_report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n📊 Progressive Fine-tuned Model Evaluation Results:")
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"📈 Recall: {recall:.4f}")
        print(f"⭐ F1 Score: {f1:.4f}")
        print(f"\n📋 Classification Report:")
        print(class_report)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix - Progressive Fine-tuned Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'probabilities': all_probabilities,
            'true_labels': all_labels
        }

def create_multi_dataset_loaders(dataset_configs, batch_size=16, balance_dataset=True, model_name='xception'):
    """
    Create data loaders from multiple datasets
    Args:
        dataset_configs: List of tuples (dataset_name, base_dir)
        batch_size: Batch size for data loaders
        balance_dataset: Whether to balance real and fake samples
        model_name: Model name for appropriate transforms
    """
    
    # Define possible validation directory names
    val_dirs = ['valid', 'val', 'validation']
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # Get transforms based on model
    trainer = ProgressiveFineTuneTrainer(model_name=model_name)
    train_transform = trainer.get_transforms(augment=True)
    val_transform = trainer.get_transforms(augment=False)
    
    for dataset_name, base_dir in dataset_configs:
        print(f"\n📁 Processing dataset: {dataset_name}")
        base_dir = os.path.normpath(base_dir)
        
        # Find validation directory
        val_dir = None
        for vd in val_dirs:
            potential_val = os.path.join(base_dir, vd)
            if os.path.exists(potential_val):
                val_dir = vd
                break
        
        if val_dir is None:
            print(f"❌ Could not find validation directory for {dataset_name}")
            continue
        
        # Define paths
        train_real_dir = os.path.join(base_dir, 'train', 'real')
        train_fake_dir = os.path.join(base_dir, 'train', 'fake')
        val_real_dir = os.path.join(base_dir, val_dir, 'real')
        val_fake_dir = os.path.join(base_dir, val_dir, 'fake')
        test_real_dir = os.path.join(base_dir, 'test', 'real')
        test_fake_dir = os.path.join(base_dir, 'test', 'fake')
        
        # Check if directories exist
        directories = {
            'train_real': train_real_dir,
            'train_fake': train_fake_dir,
            'val_real': val_real_dir,
            'val_fake': val_fake_dir,
            'test_real': test_real_dir,
            'test_fake': test_fake_dir
        }
        
        missing_dirs = []
        for name, dir_path in directories.items():
            if not os.path.exists(dir_path):
                missing_dirs.append((name, dir_path))
        
        if missing_dirs:
            print(f"⚠️  Missing {len(missing_dirs)} directories in {dataset_name}:")
            for name, path in missing_dirs:
                print(f"   - {name}: {path}")
            continue
        
        # Create datasets for this dataset
        train_dataset = DeepfakeDataset(train_real_dir, train_fake_dir, transform=train_transform, balance_dataset=balance_dataset)
        val_dataset = DeepfakeDataset(val_real_dir, val_fake_dir, transform=val_transform, balance_dataset=balance_dataset)
        test_dataset = DeepfakeDataset(test_real_dir, test_fake_dir, transform=val_transform, balance_dataset=False)
        
        if len(train_dataset) > 0:
            train_datasets.append(train_dataset)
        if len(val_dataset) > 0:
            val_datasets.append(val_dataset)
        if len(test_dataset) > 0:
            test_datasets.append(test_dataset)
        
        print(f"✅ {dataset_name}: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    if not train_datasets:
        print("❌ No valid training datasets found!")
        return None, None, None
    
    # Combine all datasets
    from torch.utils.data import ConcatDataset
    
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets) if val_datasets else None
    combined_test = ConcatDataset(test_datasets) if test_datasets else None
    
    print(f"\n📊 Combined Dataset Statistics:")
    print(f"   Training: {len(combined_train)} samples from {len(train_datasets)} datasets")
    print(f"   Validation: {len(combined_val) if combined_val else 0} samples from {len(val_datasets)} datasets")
    print(f"   Test: {len(combined_test) if combined_test else 0} samples from {len(test_datasets)} datasets")
    
    # Create data loaders
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) if combined_val else None
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) if combined_test else None
    
    return train_loader, val_loader, test_loader

def progressive_fine_tune_fourth_dataset():
    """Main function for progressive fine-tuning with fourth dataset"""
    
    # Configuration for fourth dataset fine-tuning
    config = {
        # Define all four datasets
        'datasets': [
            ('real vs fake', 'archive/real_vs_fake/real-vs-fake'),  # UPDATE THESE PATHS
            ('FACES', 'Dataset_deepfake'),
            ('Celeb_DF', 'CelebDF_Frames'),
            ('Face_Forensics', 'FFPP_Frames')  # Fourth dataset
        ],
        'pretrained_model_path': 'progressive_fine_tuned_model/2nd_tuned_xception_model.pth',
        'model_name': 'xception',
        'epochs': 25,  # Slightly more epochs for combined dataset
        'batch_size': 16,
        'learning_rate': 3e-6,  # Even lower learning rate
        'weight_decay': 1e-5,
        'save_dir': 'fourth_dataset_fine_tuned_model',
        'patience': 10,
        'balance_dataset': True,
        'fine_tune_strategy': 'progressive'
    }
    
    print("🔄 Progressive Fine-tuning with Fourth Dataset")
    print("=" * 60)
    print("🎯 Fourth Dataset Fine-tuning Strategy:")
    print("   - Combining all four datasets")
    print("   - Very low learning rate (3e-6)")
    print("   - Progressive layer-wise learning rates")
    print("   - Extended training with patience")
    print("=" * 60)
    
    # Check if datasets exist
    available_datasets = []
    for dataset_name, dataset_path in config['datasets']:
        if os.path.exists(dataset_path):
            available_datasets.append((dataset_name, dataset_path))
            print(f"✅ Found dataset: {dataset_name}")
        else:
            print(f"❌ Dataset not found: {dataset_name} at {dataset_path}")
    
    if len(available_datasets) < 2:
        print("❌ Need at least 2 datasets to proceed!")
        return
    
    config['datasets'] = available_datasets
    
    # Check if pre-trained model exists
    if not os.path.exists(config['pretrained_model_path']):
        print(f"⚠️ Pre-trained model not found: {config['pretrained_model_path']}")
        print("🔍 Looking for alternative models...")
        
        # Try to find the most recent model
        alternative_paths = [
            'progressive_fine_tuned_model/2nd_tuned_xception_model.pth',
            'fine_tuned_xception_model/best_fine_tuned_model.pth',
            'xception_deepfake_model/best_face_model.pth'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                config['pretrained_model_path'] = alt_path
                print(f"✅ Found alternative model: {alt_path}")
                break
        else:
            print("❌ No pre-trained models found. Starting from scratch.")
            config['pretrained_model_path'] = None
    
    # Prepare multi-dataset data loaders
    print(f"📦 Loading multiple datasets...")
    train_loader, val_loader, test_loader = create_multi_dataset_loaders(
        config['datasets'],
        batch_size=config['batch_size'],
        balance_dataset=config['balance_dataset'],
        model_name=config['model_name']
    )
    
    if train_loader is None:
        print("❌ Failed to create data loaders. Please check your dataset structure.")
        return
    
    # Initialize trainer with pre-trained model
    print(f"🤖 Initializing {config['model_name']} model for fourth dataset fine-tuning...")
    trainer = ProgressiveFineTuneTrainer(
        model_name=config['model_name'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        pretrained_path=config['pretrained_model_path']
    )
    
    # Setup optimizer for progressive fine-tuning
    trainer.setup_optimizer(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        fine_tune_strategy=config['fine_tune_strategy']
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"🎮 GPU Memory: {gpu_memory:.1f} GB total, {allocated_memory:.1f} GB allocated")
    
    # Start progressive fine-tuning with fourth dataset
    print("🎯 Starting fourth dataset fine-tuning...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=config['epochs'],
        save_dir=config['save_dir'],
        patience=config['patience']
    )
    
    # Plot training history
    print("📈 Plotting fourth dataset fine-tuning history...")
    trainer.plot_training_history(os.path.join(config['save_dir'], 'fourth_dataset_training_history.png'))
    
    # Evaluate on test set
    if test_loader:
        print("🧪 Evaluating fourth dataset fine-tuned model on test set...")
        evaluation_results = trainer.evaluate_model(
            test_loader,
            save_path=os.path.join(config['save_dir'], 'fourth_dataset_confusion_matrix.png')
        )
    else:
        print("⚠️ No test loader available for evaluation")
        evaluation_results = {'accuracy': 0, 'f1_score': 0, 'precision': 0, 'recall': 0}
    
    # Save fourth dataset fine-tuning configuration and results
    results = {
        'config': config,
        'final_metrics': {
            'test_accuracy': float(evaluation_results['accuracy']),
            'test_precision': float(evaluation_results['precision']),
            'test_recall': float(evaluation_results['recall']),
            'test_f1': float(evaluation_results['f1_score'])
        },
        'dataset_statistics': {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset) if val_loader else 0,
            'test_samples': len(test_loader.dataset) if test_loader else 0,
            'datasets_used': [name for name, path in config['datasets']]
        },
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': config['model_name'],
            'pretrained_model': config['pretrained_model_path'],
            'fine_tune_stage': 'fourth_dataset'
        },
        'training_history': history,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(config['save_dir'], 'fourth_dataset_fine_tuning_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Fourth Dataset Fine-tuning Completed!")
    print(f"💾 Fourth dataset fine-tuned model saved in: {config['save_dir']}")
    print(f"📊 Final Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"📊 Final Test F1 Score: {evaluation_results['f1_score']:.4f}")
    print(f"🚀 Your model is now highly specialized on four datasets and ready for deployment!")

def compare_all_models_with_fourth():
    """Compare performance across all training stages including fourth dataset"""
    print("\n📊 Comparing All Model Versions (Including Fourth Dataset)")
    print("=" * 80)
    
    model_versions = [
        ('Original', 'xception_deepfake_model/best_face_model.pth'),
        ('First Fine-tune', 'fine_tuned_xception_model/best_fine_tuned_model.pth'),
        ('Progressive Fine-tune', 'progressive_fine_tuned_model/2nd_tuned_xception_model.pth'),
        ('Fourth Dataset', 'fourth_dataset_fine_tuned_model/3rd_tuned_xception_model.pth')
    ]
    
    # Load test data from fourth dataset
    print("📦 Loading test data for comparison...")
    _, _, test_loader = create_multi_dataset_loaders(
        [('Fourth_Dataset', 'FFPP_Frames')],  # UPDATE THIS PATH
        batch_size=16,
        model_name='xception'
    )
    
    if test_loader is None:
        print("❌ Could not load test data for comparison")
        return
    
    results = {}
    
    for version_name, model_path in model_versions:
        if os.path.exists(model_path):
            print(f"\n🔍 Testing {version_name} model...")
            trainer = ProgressiveFineTuneTrainer(
                model_name='xception',
                pretrained_path=model_path
            )
            results[version_name] = trainer.evaluate_model(test_loader)
        else:
            print(f"⚠️ Model not found: {model_path}")
    
    # Print comparison table
    print("\n📈 Performance Comparison Across All Stages:")
    print("=" * 100)
    print(f"{'Model Version':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Improvement':<12}")
    print("-" * 100)
    
    baseline_accuracy = None
    for version_name in ['Original', 'First Fine-tune', 'Progressive Fine-tune', 'Fourth Dataset']:
        if version_name in results:
            metrics = results[version_name]
            accuracy = metrics['accuracy']
            
            if baseline_accuracy is None:
                baseline_accuracy = accuracy
                improvement = 0.0
            else:
                improvement = accuracy - baseline_accuracy
            
            print(f"{version_name:<25} | {accuracy:.4f}     | {metrics['precision']:.4f}     | {metrics['recall']:.4f}     | {metrics['f1_score']:.4f}     | {improvement:+.4f}")

if __name__ == "__main__":
    # Run fourth dataset fine-tuning
    progressive_fine_tune_fourth_dataset()
    
    # Compare all model versions including fourth dataset
    print("\n" + "="*100)
    compare_all_models_with_fourth()