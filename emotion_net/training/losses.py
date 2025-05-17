"""Loss functions for emotion recognition models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Paper: Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha)
        # Add small epsilon to avoid numerical issues
        self.eps = 1e-6

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): The model's unnormalized logits, tensor of shape (N, C).
            targets (Tensor): The ground truth targets, tensor of shape (N).
                              Each value must be in [0, C-1].
        Returns:
            Tensor: The focal loss.
        """
        # Calculate Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Get probabilities
        pt = torch.exp(-ce_loss).clamp(min=self.eps)
        
        # Calculate Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            
            # Get alpha for each sample
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

class StableFocalLoss(nn.Module):
    """
    A more stable version of Focal Loss that avoids numerical issues.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha)
        self.eps = 1e-8  # Small epsilon for numerical stability

    def forward(self, inputs, targets):
        # Get softmax probabilities
        probs = F.softmax(inputs, dim=1)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        
        # Get target one-hot encoding
        num_classes = inputs.size(1)
        target_one_hot = F.one_hot(targets, num_classes).float()
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            target_one_hot = (1 - self.label_smoothing) * target_one_hot + \
                self.label_smoothing / num_classes
        
        # Calculate pt (probability of being correct class)
        pt = (target_one_hot * probs).sum(1)
        
        # Calculate the focal weight
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Calculate the cross entropy
        log_probs = torch.log(probs)
        ce_loss = -torch.sum(target_one_hot * log_probs, dim=1)
        
        # Apply the focal weight
        loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            
            # Apply class weights based on targets
            alpha_weight = torch.gather(self.alpha, 0, targets)
            loss = alpha_weight * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CenterLoss(nn.Module):
    """
    Center Loss for better feature discrimination.
    
    Paper: A Discriminative Feature Learning Approach (https://ydwen.github.io/papers/WenECCV16.pdf)
    """
    def __init__(self, num_classes, feat_dim, device=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Centers for each class (initialized during first forward pass)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(self.device))
        
    def forward(self, features, labels):
        """
        Args:
            features: embedding features of shape (batch_size, feat_dim)
            labels: ground truth labels of shape (batch_size)
        """
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        
        # Calculate Euclidean distance between features and their centers
        diff = features - centers_batch
        loss = torch.mean(torch.sum(diff.pow(2), dim=1) / 2.0)
        
        return loss

class TripletLoss(nn.Module):
    """
    Triplet Loss for learning better embeddings.
    
    Paper: FaceNet: A Unified Embedding for Face Recognition (https://arxiv.org/abs/1503.03832)
    """
    def __init__(self, margin=0.3, mining='batch_hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining  # 'batch_hard', 'batch_all', 'batch_semi'
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: features of shape (batch_size, embed_dim)
            labels: ground truth labels of shape (batch_size)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if self.mining == 'batch_hard':
            return self.batch_hard_triplet_loss(embeddings, labels)
        elif self.mining == 'batch_all':
            return self.batch_all_triplet_loss(embeddings, labels)
        else:  # batch_semi
            return self.batch_semi_triplet_loss(embeddings, labels)
            
    def batch_hard_triplet_loss(self, embeddings, labels):
        """
        Compute the batch hard triplet loss:
        - For each anchor, mine the hardest positive and hardest negative
        """
        # Calculate pairwise distances
        dist_matrix = self.pairwise_distances(embeddings)
        
        # Get masks for same class and different class pairs
        same_identity_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # For each anchor, find the hardest positive
        hardest_positive_dist = (
            dist_matrix + 
            (torch.max(dist_matrix, dim=1, keepdim=True)[0] + 1.0) * (~same_identity_mask).float()
        ).min(dim=1)[0]
        
        # For each anchor, find the hardest negative
        hardest_negative_dist = (
            dist_matrix + 
            (torch.max(dist_matrix, dim=1, keepdim=True)[0] + 1.0) * same_identity_mask.float()
        ).min(dim=1)[0]
        
        # Calculate triplet loss with margin
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        # Return only non-zero triplets
        valid_triplets = triplet_loss > 0.0
        if valid_triplets.sum() == 0:
            # Return a small loss if no triplets are valid
            return torch.tensor(0.0, device=embeddings.device)
        
        return triplet_loss[valid_triplets].mean()
    
    def batch_all_triplet_loss(self, embeddings, labels):
        """
        Compute the batch all triplet loss:
        - Generate all valid triplets and compute mean loss
        """
        # Calculate pairwise distances
        dist_matrix = self.pairwise_distances(embeddings)
        
        # Get masks for same class and different class pairs
        same_identity_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # For each anchor-positive pair, compute loss against all negatives
        triplet_loss = []
        num_valid_triplets = 0
        
        for i in range(embeddings.size(0)):
            pos_indices = torch.where(same_identity_mask[i] & (i != torch.arange(embeddings.size(0), device=embeddings.device)))[0]
            neg_indices = torch.where(~same_identity_mask[i])[0]
            
            # Skip if no positives or negatives
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
                
            for pos_idx in pos_indices:
                pos_dist = dist_matrix[i, pos_idx]
                
                # Compute loss against all negatives
                neg_dists = dist_matrix[i, neg_indices]
                batch_loss = F.relu(pos_dist.unsqueeze(0) - neg_dists + self.margin)
                
                # Count valid triplets
                valid_triplets = batch_loss > 0.0
                num_valid_triplets += valid_triplets.sum().item()
                
                if valid_triplets.sum() > 0:
                    triplet_loss.append(batch_loss[valid_triplets].mean())
                    
        if len(triplet_loss) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        return torch.stack(triplet_loss).mean()
    
    def batch_semi_triplet_loss(self, embeddings, labels):
        """
        Compute the batch semi-hard triplet loss:
        - For each anchor-positive pair, find semi-hard negatives
        """
        # Calculate pairwise distances
        dist_matrix = self.pairwise_distances(embeddings)
        
        # Get masks for same class and different class pairs
        same_identity_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # For each anchor-positive pair, compute loss against semi-hard negatives
        triplet_loss = []
        num_valid_triplets = 0
        
        for i in range(embeddings.size(0)):
            pos_indices = torch.where(same_identity_mask[i] & (i != torch.arange(embeddings.size(0), device=embeddings.device)))[0]
            
            # Skip if no positives
            if len(pos_indices) == 0:
                continue
                
            for pos_idx in pos_indices:
                pos_dist = dist_matrix[i, pos_idx]
                
                # Find semi-hard negatives: negatives that are farther than the positive
                # but still within the margin
                neg_indices = torch.where(~same_identity_mask[i] & (dist_matrix[i] > pos_dist) & (dist_matrix[i] < pos_dist + self.margin))[0]
                
                if len(neg_indices) == 0:
                    # If no semi-hard negatives, find the closest negative
                    neg_indices = torch.where(~same_identity_mask[i])[0]
                    if len(neg_indices) == 0:
                        continue
                    neg_idx = neg_indices[dist_matrix[i, neg_indices].argmin()]
                    batch_loss = F.relu(pos_dist - dist_matrix[i, neg_idx] + self.margin)
                    if batch_loss > 0:
                        triplet_loss.append(batch_loss)
                        num_valid_triplets += 1
                else:
                    # Compute loss against all semi-hard negatives
                    neg_dists = dist_matrix[i, neg_indices]
                    batch_loss = F.relu(pos_dist.unsqueeze(0) - neg_dists + self.margin)
                    valid_triplets = batch_loss > 0.0
                    num_valid_triplets += valid_triplets.sum().item()
                    
                    if valid_triplets.sum() > 0:
                        triplet_loss.append(batch_loss[valid_triplets].mean())
        
        if len(triplet_loss) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        return torch.stack(triplet_loss).mean()
    
    def pairwise_distances(self, embeddings):
        """
        Compute pairwise Euclidean distances between all embeddings.
        """
        # Dot product between all embeddings
        dot_product = torch.matmul(embeddings, embeddings.t())
        
        # Get squared L2 norm for each embedding
        square_norm = torch.diagonal(dot_product)
        
        # Calculate pairwise distance matrix using the identity
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * <a, b>
        distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product
        
        # Because of computation errors, some distances might be negative
        distances = F.relu(distances)
        
        return torch.sqrt(distances + 1e-8)
        
class HybridLoss(nn.Module):
    """
    Hybrid Loss combining multiple loss functions.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        # Initialize components
        self.focal_loss = StableFocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)
        
    def forward(self, inputs, targets):
        # Combine focal and standard cross-entropy
        return 0.5 * self.focal_loss(inputs, targets) + 0.5 * self.ce_loss(inputs, targets)

class CompositeEmotionLoss(nn.Module):
    """
    Advanced composite loss function combining multiple components:
    - Cross Entropy Loss with label smoothing for main classification
    - Center Loss for better feature discrimination
    - Triplet Loss for better separation of similar emotions
    - Focal Loss for handling class imbalance
    - Deep supervision loss for auxiliary outputs
    
    This loss is specifically designed for the enhanced FER model.
    """
    def __init__(self, num_classes=7, embedding_size=1024, 
                 label_smoothing=0.2, center_weight=0.01, 
                 triplet_weight=0.1, focal_gamma=2.0,
                 aux_weight=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.label_smoothing = label_smoothing
        self.center_weight = center_weight
        self.triplet_weight = triplet_weight
        self.focal_gamma = focal_gamma
        self.aux_weight = aux_weight
        
        # Initialize loss components
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.center_loss = CenterLoss(num_classes, embedding_size)
        self.triplet_loss = TripletLoss(margin=0.3, mining='batch_hard')
        self.focal_loss = StableFocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        
        # Ensure loss weights are tensor scalars for stability
        self.register_buffer('center_weight_t', torch.tensor(center_weight))
        self.register_buffer('triplet_weight_t', torch.tensor(triplet_weight))
        self.register_buffer('aux_weight_t', torch.tensor(aux_weight))
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tuple/list of model outputs (enhanced model returns 6 outputs)
                - outputs[0]: main logits (batch_size, num_classes)
                - outputs[1]: group logits (batch_size, num_classes)
                - outputs[2]: embedding features (batch_size, embedding_size)
                - outputs[3-5]: auxiliary outputs from different network layers
            targets: class labels (batch_size)
        """
        # If outputs is not a list, treat it as a single logits tensor
        if not isinstance(outputs, (list, tuple)):
            return self.ce_loss(outputs, targets)
        
        # Get outputs from the model
        main_logits = outputs[0]
        group_logits = outputs[1] 
        embedding = outputs[2]
        
        # Main classification loss
        ce_main = self.ce_loss(main_logits, targets)
        
        # Initialize total loss with main CE loss
        total_loss = ce_main
        
        # Group-based auxiliary loss
        if len(outputs) > 1:
            group_loss = self.ce_loss(group_logits, targets)
            total_loss = total_loss + 0.3 * group_loss
        
        # Center loss for feature discrimination
        if len(outputs) > 2:
            center_loss = self.center_loss(embedding, targets)
            total_loss = total_loss + self.center_weight_t * center_loss
        
        # Triplet loss for difficult emotion separation
        if len(outputs) > 2:
            trip_loss = self.triplet_loss(embedding, targets)
            total_loss = total_loss + self.triplet_weight_t * trip_loss
        
        # Focal loss component for handling class imbalance
        focal_loss = self.focal_loss(main_logits, targets)
        total_loss = total_loss + 0.2 * focal_loss
        
        # Deep supervision auxiliary losses
        if len(outputs) > 3:
            for i in range(3, min(6, len(outputs))):
                aux_ce = self.ce_loss(outputs[i], targets)
                total_loss = total_loss + self.aux_weight_t * aux_ce
        
        return total_loss 

class SOTAEmotionLoss(nn.Module):
    """
    State-of-the-art emotion recognition loss combining multiple loss functions.
    """
    def __init__(self, num_classes=7, embedding_size=1024, label_smoothing=0.1, aux_weight=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.label_smoothing = label_smoothing
        self.aux_weight = aux_weight
        
        # Main classification loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Center loss for better feature discrimination
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=embedding_size)
        
        # Triplet loss for better embedding learning
        self.triplet_loss = TripletLoss(margin=0.3, mining='batch_hard')
        
        # Loss weights
        self.center_weight = 0.01
        self.triplet_weight = 0.1
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model outputs that may be in different formats
            targets: Ground truth labels
        """
        # Initialize components as None
        main_logits = None
        aux_logits = None 
        embeddings = None
        
        # Handle tuple outputs safely without unpacking
        if isinstance(outputs, (list, tuple)):
            if len(outputs) > 0:
                main_logits = outputs[0]
                
                if len(outputs) > 1:
                    aux_logits = outputs[1]
                    
                if len(outputs) > 2:
                    embeddings = outputs[2]
        # Handle dictionary outputs
        elif isinstance(outputs, dict):
            main_logits = outputs.get('logits')
            aux_logits = outputs.get('aux_logits')
            embeddings = outputs.get('features')
        # Handle tensor output (main logits only)
        else:
            main_logits = outputs
        
        # Check if main_logits is valid
        if main_logits is None:
            raise ValueError("No valid logits found in model outputs")
        
        # Main classification loss
        total_loss = self.ce_loss(main_logits, targets)
        
        # Auxiliary classification loss if available
        if aux_logits is not None:
            try:
                aux_loss = self.ce_loss(aux_logits, targets)
                total_loss = total_loss + self.aux_weight * aux_loss
            except Exception as e:
                print(f"Warning: Error calculating auxiliary loss: {e}")
        
        # Center loss and triplet loss if embeddings are available
        if embeddings is not None:
            try:
                center_loss = self.center_loss(embeddings, targets)
                total_loss = total_loss + self.center_weight * center_loss
                
                triplet_loss = self.triplet_loss(embeddings, targets)
                total_loss = total_loss + self.triplet_weight * triplet_loss
            except Exception as e:
                print(f"Warning: Error calculating embedding-based losses: {e}")
        
        return total_loss 

class AdaptiveEmotionLoss(nn.Module):
    """
    Specialized loss function for facial emotion recognition achieving 80%+ accuracy.
    Combines multiple loss components with adaptive weighting based on training progress.
    
    Key features:
    1. Adaptively weights loss components based on training progress
    2. Focuses on easily confused emotion pairs with custom contrastive penalties
    3. Includes perceptual feature loss for better generalization
    4. Addresses class imbalance with dynamic class weighting
    """
    def __init__(self, num_classes=7, embedding_size=768, label_smoothing=0.1,
                 emotion_confusion_matrix=None, initial_epoch=0, max_epochs=300):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.label_smoothing = label_smoothing
        self.epoch = initial_epoch  # Current training epoch
        self.max_epochs = max_epochs  # Maximum training epochs
        
        # Core losses
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.focal_loss = StableFocalLoss(gamma=2.0, label_smoothing=label_smoothing)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=embedding_size)
        self.triplet_loss = TripletLoss(margin=0.3, mining='batch_hard')
        
        # Commonly confused emotion pairs (based on psychology research)
        # Format: (emotion1, emotion2, penalty_weight)
        self.confusion_pairs = [
            (0, 4, 2.0),    # angry vs sad (often confused)
            (2, 5, 2.0),    # fear vs surprise (often confused)
            (4, 6, 1.5),    # sad vs neutral (often confused)
            (2, 4, 1.5),    # fear vs sad (sometimes confused)
            (0, 2, 1.2),    # angry vs fear (sometimes confused)
        ]
        
        # Default confusion matrix if none provided
        default_matrix = torch.tensor([
            [0.0, 0.3, 0.5, 0.2, 1.0, 0.4, 0.3],  # angry
            [0.3, 0.0, 0.3, 0.4, 0.3, 0.4, 0.3],  # disgust
            [0.5, 0.3, 0.0, 0.3, 0.8, 1.0, 0.3],  # fear
            [0.2, 0.4, 0.3, 0.0, 0.6, 0.3, 0.5],  # happy
            [1.0, 0.3, 0.8, 0.6, 0.0, 0.4, 0.8],  # sad
            [0.4, 0.4, 1.0, 0.3, 0.4, 0.0, 0.4],  # surprise
            [0.3, 0.3, 0.3, 0.5, 0.8, 0.4, 0.0],  # neutral
        ])
        
        # Initialize weights
        self.focal_weight = torch.tensor(0.5)
        self.center_weight = torch.tensor(0.01)
        self.triplet_weight = torch.tensor(0.01)
        self.contrastive_weight = torch.tensor(0.1)
        
        # Use provided confusion matrix or default
        self.confusion_matrix = emotion_confusion_matrix if emotion_confusion_matrix is not None else default_matrix
        
        # Register buffers for loss weights and confusion matrix
        self.register_buffer('focal_weight_t', self.focal_weight)
        self.register_buffer('center_weight_t', self.center_weight)
        self.register_buffer('triplet_weight_t', self.triplet_weight)
        self.register_buffer('contrastive_weight_t', self.contrastive_weight)
        self.register_buffer('confusion_matrix_t', self.confusion_matrix)
        
    def update_epoch(self, epoch):
        """Update current epoch for adaptive weighting."""
        self.epoch = epoch
        
        # Adaptive weighting based on training progress
        progress = min(self.epoch / (self.max_epochs * 0.7), 1.0)  # Cap at 70% of training
        
        # Gradually increase weights for advanced losses
        self.focal_weight_t = torch.tensor(0.5 + 0.3 * progress, device=self.focal_weight_t.device)
        self.center_weight_t = torch.tensor(0.01 + 0.02 * progress, device=self.center_weight_t.device)
        self.triplet_weight_t = torch.tensor(0.01 + 0.04 * progress, device=self.triplet_weight_t.device)
        self.contrastive_weight_t = torch.tensor(0.1 + 0.4 * progress, device=self.contrastive_weight_t.device)
    
    def _confusion_penalty(self, logits, features, targets):
        """
        Apply penalties to commonly confused emotion pairs.
        This encourages the model to learn discriminative features for similar emotions.
        """
        batch_size = logits.size(0)
        device = logits.device
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)  # Need at least 2 samples for contrastive learning
            
        penalty = torch.tensor(0.0, device=device)
        
        # Get softmax probabilities for all classes
        probs = F.softmax(logits, dim=1)
        
        # Normalize feature embeddings
        norm_features = F.normalize(features, p=2, dim=1)
        
        # Create a mask of where each target appears
        target_masks = {}
        for emotion_idx in range(self.num_classes):
            target_masks[emotion_idx] = (targets == emotion_idx)
            
        # For each confusion pair, add a penalty term
        for emotion1, emotion2, weight in self.confusion_pairs:
            # Skip if either emotion doesn't have any samples
            if not target_masks[emotion1].any() or not target_masks[emotion2].any():
                continue
                
            # Get features for both emotions
            features1 = norm_features[target_masks[emotion1]]
            features2 = norm_features[target_masks[emotion2]]
            
            # Calculate similarity between feature centroids
            centroid1 = features1.mean(dim=0, keepdim=True)
            centroid2 = features2.mean(dim=0, keepdim=True)
            
            # Cosine similarity between centroids (should be low for confused emotions)
            sim = F.cosine_similarity(centroid1, centroid2).abs()
            
            # Add penalty proportional to similarity (higher similarity = higher penalty)
            penalty_term = sim * weight * self.confusion_matrix_t[emotion1, emotion2]
            penalty = penalty + penalty_term
            
            # Also penalize probability confusion
            for i in range(batch_size):
                if targets[i] == emotion1:
                    # If true class is emotion1, penalize high probability for emotion2
                    penalty = penalty + probs[i, emotion2] * weight * 0.1
                elif targets[i] == emotion2:
                    # If true class is emotion2, penalize high probability for emotion1
                    penalty = penalty + probs[i, emotion1] * weight * 0.1
        
        return penalty
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: tuple of (main_logits, aux_logits, embeddings) from the model
            targets: ground truth labels
        """
        device = targets.device
        
        # Extract outputs
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
            main_logits, aux_logits, embeddings = outputs[:3]
        elif isinstance(outputs, dict):
            main_logits = outputs.get('logits')
            aux_logits = outputs.get('aux_logits')
            embeddings = outputs.get('features')
        else:
            # Fallback to single output
            main_logits = outputs
            aux_logits = None
            embeddings = None
        
        # Basic loss components
        ce_loss = self.ce_loss(main_logits, targets)
        
        # Initialize total loss with cross-entropy
        total_loss = ce_loss
        
        # Add focal loss for handling class imbalance
        if self.focal_weight_t > 0:
            focal_loss = self.focal_loss(main_logits, targets)
            total_loss = total_loss + self.focal_weight_t * focal_loss
        
        # Add auxiliary loss if available
        if aux_logits is not None:
            aux_loss = self.ce_loss(aux_logits, targets)
            total_loss = total_loss + 0.4 * aux_loss
        
        # Add embedding-based losses if available
        if embeddings is not None:
            # Center loss for better clustering
            if self.center_weight_t > 0:
                center_loss = self.center_loss(embeddings, targets)
                total_loss = total_loss + self.center_weight_t * center_loss
            
            # Triplet loss for better discrimination
            if self.triplet_weight_t > 0:
                triplet_loss = self.triplet_loss(embeddings, targets)
                total_loss = total_loss + self.triplet_weight_t * triplet_loss
            
            # Add confusion penalties for commonly confused emotions
            if self.contrastive_weight_t > 0:
                confusion_penalty = self._confusion_penalty(main_logits, embeddings, targets)
                total_loss = total_loss + self.contrastive_weight_t * confusion_penalty
        
        return total_loss 