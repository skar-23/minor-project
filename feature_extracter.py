"""
Multimodal Feature Extractor using 4 pre-trained encoders:
- BERT (text pattern), CLIP (text/image semantic), Swin-T (image pattern)
Outputs 4 feature streams (512-d each) for hierarchical fusion.
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, CLIPProcessor, CLIPModel
from torchvision.models import swin_t, Swin_T_Weights
from torchvision import transforms
from PIL import Image
from config import config

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Text Pattern Encoder (BERT)
        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        self.bert_tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
        
        # 2. Text & Image Semantic Encoder (CLIP)
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        
        # 3. Image Pattern Encoder (Swin-T)
        self.swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        # Return pooled 768-d features instead of 1000-d logits
        self.swin.head = nn.Identity()
        
        # Freeze pre-trained models (we're only extracting features)
        self._freeze_models()
        
        # 4. Projection networks (g^u in paper) - 2-layer MLP
        self.projection = nn.ModuleDict({
            'bert': self._make_projection(768),      # BERT hidden size
            'clip_text': self._make_projection(512), # CLIP text embedding
            'swin': self._make_projection(768),      # Swin-T output
            'clip_image': self._make_projection(512) # CLIP image embedding
        })
        
        # Image transform for Swin-T (CLIP uses its own processor)
        self.swin_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _freeze_models(self):
        """Freeze pre-trained models"""
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.swin.parameters():
            param.requires_grad = False
    
    def _make_projection(self, input_dim):
        """Create 2-layer MLP projection network"""
        return nn.Sequential(
            nn.Linear(input_dim, config.FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM)
        )
    
    def extract_features(self, texts, images):
        """
        Extract all 4 feature types for a batch
        
        Returns: dict with keys:
            - text_pattern: BERT features [batch, 512]
            - text_semantic: CLIP text features [batch, 512]
            - image_pattern: Swin-T features [batch, 512]
            - image_semantic: CLIP image features [batch, 512]
        """
        features = {}
        
        # Move to device
        device = next(self.parameters()).device
        
        # 1. Text Pattern (BERT)
        bert_inputs = self.bert_tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            bert_output = self.bert(**bert_inputs)
            q = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
            features['text_pattern'] = self.projection['bert'](q)
        
        # 2. Text Semantic (CLIP Text)
        clip_text_inputs = self.clip_processor(
            text=texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            clip_text_output = self.clip.get_text_features(**clip_text_inputs)
            features['text_semantic'] = self.projection['clip_text'](clip_text_output)
        
        # 3. Image Pattern (Swin-T)
        swin_images = torch.stack([self.swin_transform(img) for img in images]).to(device)
        with torch.no_grad():
            swin_output = self.swin(swin_images)
            features['image_pattern'] = self.projection['swin'](swin_output)
        
        # 4. Image Semantic (CLIP Image)
        # CLIP expects PIL images, process in batches
        clip_image_inputs = self.clip_processor(
            images=images, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            clip_image_output = self.clip.get_image_features(**clip_image_inputs)
            features['image_semantic'] = self.projection['clip_image'](clip_image_output)
        
        return features
    
    def forward(self, texts, images):
        """Alias for extract_features"""
        return self.extract_features(texts, images)