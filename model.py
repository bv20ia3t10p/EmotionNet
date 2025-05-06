import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from config import *
import timm # type: ignore
import math
import os


# Gradient Accumulation Steps - to handle large batch sizes effectively
# Use environment variable if provided, otherwise calculate based on batch size
ACCUMULATION_STEPS = int(os.environ.get('ACCUMULATION_STEPS', max(1, BATCH_SIZE // 512)))


# Memory efficient attention module
class EfficientAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(EfficientAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.query = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Reshape for efficient computation
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        
        # Compute attention using matrix multiplication
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Efficient value projection
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        # Reshape back to the original format
        out = out.view(batch_size, C, height, width)
        
        # Residual connection with learnable parameter
        out = self.gamma * out + x
        return out


# Simplified Attention Module for better early training
class SimpleAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SimpleAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Add max pooling for better feature focus
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),  # Process both avg and max features
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        att = self.fc(torch.cat([avg_out, max_out], dim=1)).view(b, c, 1, 1)
        return x * att


class CBAM(nn.Module):
    def __init__(self, channels, reduction=4, kernel_size=7):  # Reduced reduction factor for stronger attention
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_dim = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()


    def forward(self, x):
        b, c, h, w = x.size()
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att.expand_as(x)
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_att))
        x = x * spatial_att.expand_as(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for attention-based feature recalibration"""
    def __init__(self, channel, reduction=4):  # Reduced reduction factor for stronger attention
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)  # Add SE block
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_ch)
            )
            
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE block
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


# Improved ECA attention module (Efficient Channel Attention)
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# High-performance emotion recognition model with pretrained backbone
class AdvancedEmoteNet(nn.Module):
    def __init__(self, backbone="efficientnet_b2", pretrained=True):
        super(AdvancedEmoteNet, self).__init__()
        from config import NUM_CLASSES, HEAD_DROPOUT, FEATURE_DROPOUT, IMAGE_SIZE
        
        # Safely check if torch.cuda.is_available() to enable memory optimizations
        if torch.cuda.is_available():
            # Use channels_last memory format for better GPU performance on CUDA
            self.channels_last = True
        else:
            self.channels_last = False
        
        # Load pretrained backbone with ImageNet weights
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''   # Remove pooling
        )
        
        # Enable gradient checkpointing to reduce memory usage
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        # Get backbone output features
        if 'efficientnetv2' in backbone:
            # EfficientNetV2 has different feature dimensions
            if 's' in backbone:
                feature_dim = 1280
            elif 'm' in backbone:
                feature_dim = 1280
            elif 'l' in backbone:
                feature_dim = 1280
            else:
                feature_dim = 1408
            self.features_only = False
        elif 'efficientnet' in backbone:
            if 'b0' in backbone:
                feature_dim = 1280
            elif 'b1' in backbone:
                feature_dim = 1280
            elif 'b2' in backbone:
                feature_dim = 1408
            elif 'b3' in backbone:
                feature_dim = 1536
            elif 'b4' in backbone:
                feature_dim = 1792
            else:
                feature_dim = 2048  # For b5, b6, b7
            self.features_only = False
        elif 'resnet' in backbone:
            feature_dim = 2048  # ResNet feature dimension
            self.features_only = False
        else:
            feature_dim = 1280  # Default feature dimension
            self.features_only = False
            
        # Add attention modules for better feature focus - using simpler attention for stability
        self.attention = SimpleAttention(feature_dim, reduction=16)
        
        # Add extra layers for better feature refinement with dropout - simplified
        self.extra_conv = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(FEATURE_DROPOUT)
        )
        
        # Global pooling with multi-scale feature extraction
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        
        # Enhanced classifier with deeper architecture
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT * 0.8),
            
            nn.Linear(256, NUM_CLASSES)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Only initialize the weights of the new layers we added, not the pretrained backbone
        for m in [self.attention, self.extra_conv, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    # Use specialized initialization based on layer position
                    if layer == list(self.classifier.modules())[-1]:  # Final classification layer
                        # Use smaller initialization for final layer for better stability
                        nn.init.normal_(layer.weight, 0, 0.001)
                    else:
                        # Better initialization for hidden layers
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        # Convert to channels_last memory format if on CUDA for better performance
        if self.channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
            
        # Extract features using the backbone
        if self.features_only:
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
            
        # Apply attention for better feature focus - simplified approach
        features = self.attention(features)
        
        # Apply extra convolutional layers for refinement
        features = self.extra_conv(features)
        
        # Apply global pooling with multi-scale feature extraction
        avg_pool = self.global_avg_pool(features).view(features.size(0), -1)
        max_pool = self.global_max_pool(features).view(features.size(0), -1)
        
        # Concatenate pooling results for richer feature representation
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Add feature mixing for better robustness
        if self.training and torch.cuda.is_available():
            batch_size = x.size(0)
            if batch_size > 1:  # Only apply when batch size > 1
                # Create a random permutation of batch indices
                perm = torch.randperm(batch_size).cuda()
                # Mix a small portion of features (5%)
                mix_ratio = 0.05
                # Select mixed features based on random mask
                mask = (torch.rand(batch_size) < mix_ratio).float().cuda().view(-1, 1)
                x = x * (1 - mask) + x[perm] * mask
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


# Use Convolutional Transformer model for better feature extraction
class EmotionViT(nn.Module):
    def __init__(self, backbone="vit_base_patch16_224", pretrained=True):
        super(EmotionViT, self).__init__()
        from config import NUM_CLASSES
        
        # Safely check if torch.cuda.is_available() to enable memory optimizations
        if torch.cuda.is_available():
            # Use channels_last memory format for better GPU performance on CUDA
            self.channels_last = True
        else:
            self.channels_last = False
        
        # Use Vision Transformer as backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
        )
        
        # Enable gradient checkpointing to reduce memory usage if available
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        # Get backbone output features
        feature_dim = self.backbone.num_features

        # Add head with batch norm and dropout
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, NUM_CLASSES)
        )
        
    def forward(self, x):
        # Convert to channels_last memory format if on CUDA for better performance
        if self.channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
            
        # Extract features using the backbone
        features = self.backbone.forward_features(x)
        
        # Handle features with class token (ViT models output [batch_size, num_tokens, dim])
        # Extract just the class token which is the first token (index 0)
        if len(features.shape) == 3:
            features = features[:, 0]
            
        # Apply head
        x = self.head(features)
        return x


# Original ResEmoteNet model (kept for compatibility)
class ResEmoteNet(nn.Module):
    def __init__(self):
        super(ResEmoteNet, self).__init__()
        from config import NUM_CLASSES
        
        # Initial convolution layers with more filters
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(256)
        
        # Enhanced residual blocks - more blocks and deeper architecture
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 512, stride=1)
        self.res_block3 = ResidualBlock(512, 512, stride=1)  # Added extra residual block
        self.res_block4 = ResidualBlock(512, 768, stride=2)  # Added deeper block with more filters
        
        # Pooling and fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Combine both pooling methods with larger fully connected layers
        self.fc1 = nn.Linear(768 * 2, 512)  # Increased size
        self.fc2 = nn.Linear(512, 256)  # Increased size
        self.fc3 = nn.Linear(256, 128)  # Added extra FC layer
        
        self.dropout1 = nn.Dropout(0.3)  # Reduced dropout slightly
        self.dropout2 = nn.Dropout(0.5)  # Reduced dropout slightly
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(128)
        
        self.classifier = nn.Linear(128, NUM_CLASSES)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Initial convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.cbam(x)
        
        # Enhanced residual blocks with more depth
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)  # Added block
        x = self.res_block4(x)  # Added block
        
        # Mixed pooling (combines average and max pooling)
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        
        # Concatenate the pooling results
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Enhanced fully connected layers with batch normalization
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout2(x)
        
        x = self.classifier(x)
        return x


# Enhanced Focal Loss with label smoothing for better generalization
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        from config import NUM_CLASSES
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.device = None
        self.label_smoothing = label_smoothing
        self.num_classes = NUM_CLASSES

    def forward(self, inputs, targets):
        if self.device is None:
            self.device = inputs.device
            
        # Check if targets are one-hot encoded (happens during mixup/cutmix)
        if len(targets.shape) > 1 and targets.shape[1] == self.num_classes:
            # For mixup/cutmix, we use soft targets directly
            # Create a weighted BCE loss
            log_probs = F.log_softmax(inputs, dim=1)
            loss = -(targets * log_probs).sum(dim=1)
            
            # Apply focal weighting
            pt = torch.exp(-loss)
            focal_weight = self.alpha * (1 - pt) ** self.gamma
            return (focal_weight * loss).mean()
        else:
            # Regular case with class indices
            # Create weights tensor if provided
            if self.class_weights is not None:
                weight = torch.FloatTensor(self.class_weights).to(self.device)
                BCE_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none', label_smoothing=self.label_smoothing)
            else:
                BCE_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
                
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
            return F_loss.mean()


# Combined loss function for better training
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None, label_smoothing=0.1, kl_weight=0.1):
        super(CombinedLoss, self).__init__()
        from config import NUM_CLASSES
        self.focal_loss = FocalLoss(alpha, gamma, class_weights, label_smoothing)
        self.kl_weight = kl_weight
        self.num_classes = NUM_CLASSES
        
    def forward(self, inputs, targets):
        # Focal loss component
        focal = self.focal_loss(inputs, targets)
        
        # KL divergence component to improve regularization
        log_probs = F.log_softmax(inputs, dim=1)
        probs = F.softmax(inputs, dim=1)
        kl_div = -(log_probs * probs).sum(dim=1).mean()
        
        # Combine losses
        return focal + self.kl_weight * kl_div


# Debug helper function to check for layers without bias
def check_layers_bias(model):
    """Helper function to check if any layers in the model have None bias attributes.
    This can help identify potential issues with weight initialization.
    
    Args:
        model: The PyTorch model to check
        
    Returns:
        List of layer names that have no bias
    """
    no_bias_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            if not hasattr(module, 'bias') or module.bias is None:
                no_bias_layers.append(f"{name} ({type(module).__name__})")
    
    return no_bias_layers


# Add knowledge distillation loss for self-distillation from teacher models
class DistillationLoss(nn.Module):
    """Implements knowledge distillation loss for transferring knowledge from a teacher model."""
    def __init__(self, alpha=0.5, temperature=2.0):
        """Initialize DistillationLoss.
        
        Args:
            alpha: Weight of the distillation loss versus regular loss (0-1)
            temperature: Temperature for softening probability distributions
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels=None, criterion=None):
        """Forward pass for distillation loss.
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            labels: Ground truth labels (optional)
            criterion: Regular criterion for task-specific loss (optional)
            
        Returns:
            Combined loss of distillation and task-specific loss
        """
        # Apply temperature scaling
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Compute KL divergence loss
        distillation_loss = self.criterion(soft_student, soft_teacher) * (self.temperature ** 2)
        
        if labels is not None and criterion is not None:
            # Combine with regular task loss
            task_loss = criterion(student_logits, labels)
            return self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        else:
            # Just return distillation loss
            return distillation_loss


# Add ConvNeXt-based model for higher performance
class ConvNeXtEmoteNet(nn.Module):
    """Implements an emotion recognition model based on ConvNeXt architecture."""
    def __init__(self, backbone="convnext_base", pretrained=True):
        super(ConvNeXtEmoteNet, self).__init__()
        from config import NUM_CLASSES, HEAD_DROPOUT, FEATURE_DROPOUT, IMAGE_SIZE
        
        # Use channels_last format for better GPU performance if available
        self.channels_last = torch.cuda.is_available()
        
        # Load ConvNeXt backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''   # Remove pooling
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        # Get feature dimensions based on backbone
        if 'tiny' in backbone:
            feature_dim = 768
        elif 'small' in backbone:
            feature_dim = 768
        elif 'base' in backbone:
            feature_dim = 1024
        else:  # large
            feature_dim = 1536
        
        # Use ECA attention for better feature focus
        self.attention = ECABlock(feature_dim)
        
        # Feature refinement layer
        self.extra_conv = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),  # GELU for ConvNeXt compatibility
            nn.Dropout2d(FEATURE_DROPOUT)
        )
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        
        # Classification head with Layer Normalization (for ConvNeXt compatibility)
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT * 0.8),
            
            nn.Linear(256, NUM_CLASSES)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize the new layers we added
        for m in [self.attention, self.extra_conv, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.ones_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    # Use specialized initialization for final layer
                    if layer == list(self.classifier.modules())[-1]:
                        nn.init.normal_(layer.weight, 0, 0.001)
                    else:
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Use channels_last format for better performance
        if self.channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Extract features
        features = self.backbone(x)
        
        # Apply attention and refinement
        features = self.attention(features)
        features = self.extra_conv(features)
        
        # Global pooling
        avg_pool = self.global_avg_pool(features).view(features.size(0), -1)
        max_pool = self.global_max_pool(features).view(features.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply dropout and classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


# Add enhanced Vision Transformer model with hierarchical structure for better performance
class HierarchicalViT(nn.Module):
    """Implements a hierarchical Vision Transformer for emotion recognition."""
    def __init__(self, backbone="vit_base_patch16_224", pretrained=True):
        super(HierarchicalViT, self).__init__()
        from config import NUM_CLASSES, HEAD_DROPOUT
        
        # Use channels_last format for better GPU performance if available
        self.channels_last = torch.cuda.is_available()
        
        # Load ViT backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
        )
        
        # Enable gradient checkpointing
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Create multi-scale token pooling for hierarchical feature extraction
        self.token_pool = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU()
            ),
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU()
            ),
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU()
            )
        ])
        
        # Enhanced classifier head with hierarchical features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim // 2 * 3),  # Combined features from all levels
            nn.Linear(feature_dim // 2 * 3, 512),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT * 0.8),
            
            nn.Linear(256, NUM_CLASSES)
        )
        
        # Initialize
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize new layers
        for m in list(self.token_pool) + [self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    if layer == list(self.classifier.modules())[-1]:
                        # Final classification layer
                        nn.init.normal_(layer.weight, 0, 0.001)
                    else:
                        # Hidden layers
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def extract_hierarchical_features(self, x):
        """Extract hierarchical features from different levels of the transformer."""
        # Get all token embeddings (not just CLS token)
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        # For ViT models, tensor shape should be [batch_size, num_tokens, embedding_dim]
        if len(features.shape) == 3:
            batch_size, num_tokens, embedding_dim = features.shape
            
            # Extract tokens from different positions for hierarchical features
            # Use class token, and tokens at 1/3 and 2/3 of sequence
            cls_token = features[:, 0]  # CLS token
            mid_token = features[:, num_tokens // 3]  # Token at 1/3
            late_token = features[:, 2 * num_tokens // 3]  # Token at 2/3
            
            # Process each level with separate projection
            cls_features = self.token_pool[0](cls_token)
            mid_features = self.token_pool[1](mid_token)
            late_features = self.token_pool[2](late_token)
            
            # Combine features from different levels
            hierarchical_features = torch.cat([cls_features, mid_features, late_features], dim=1)
            return hierarchical_features
        else:
            # Fallback for non-transformer models
            return features
    
    def forward(self, x):
        # Use channels_last format for better performance
        if self.channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Extract hierarchical features
        features = self.extract_hierarchical_features(x)
        
        # Apply classification head
        x = self.classifier(features)
        
        return x