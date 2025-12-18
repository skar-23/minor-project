"""
Hierarchical Fusion Module.
Combines 4 multimodal feature streams using multi-head cross-attention.
Learns optimal fusion weights to create unified 512-d representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    """Multi-head attention module for fusing feature streams."""
    
    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, query, key, value):
        """
        Args:
            query: [batch_size, query_len, feature_dim]
            key: [batch_size, key_len, feature_dim]
            value: [batch_size, value_len, feature_dim]
        
        Returns:
            output: [batch_size, query_len, feature_dim]
            attention_weights: [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)  # [batch, query_len, feature_dim]
        K = self.key(key)      # [batch, key_len, feature_dim]
        V = self.value(value)  # [batch, value_len, feature_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.feature_dim)
        
        # Final linear projection
        output = self.fc_out(context)
        
        return output, attention_weights


class HierarchicalFusion(nn.Module):
    """
    Hierarchical Fusion Module.
    Combines 4 feature streams (text_pattern, text_semantic, image_pattern, image_semantic)
    using multi-head cross-attention mechanisms.
    """
    
    def __init__(self, feature_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Stream identifiers
        self.stream_names = ['text_pattern', 'text_semantic', 'image_pattern', 'image_semantic']
        self.num_streams = len(self.stream_names)
        
        # Create cross-attention layers for each fusion level
        self.fusion_layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadCrossAttention(feature_dim, num_heads)
                for _ in range(self.num_streams)
            ])
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(feature_dim)
                for _ in range(self.num_streams)
            ])
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim * 4, feature_dim)
                )
                for _ in range(self.num_streams)
            ])
            for _ in range(num_layers)
        ])
        
        # Final fusion layer (combine all 4 streams)
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features_dict):
        """
        Args:
            features_dict: Dictionary containing:
                - 'text_pattern': [batch_size, 512]
                - 'text_semantic': [batch_size, 512]
                - 'image_pattern': [batch_size, 512]
                - 'image_semantic': [batch_size, 512]
        
        Returns:
            fused_features: [batch_size, 512] - Unified representation
            attention_weights: Dictionary of attention matrices for visualization
        """
        batch_size = features_dict['text_pattern'].shape[0]
        
        # Initialize feature streams (add temporal dimension for attention)
        streams = []
        for name in self.stream_names:
            # [batch_size, 512] -> [batch_size, 1, 512]
            streams.append(features_dict[name].unsqueeze(1))
        
        attention_weights_list = []
        
        # Multi-layer fusion
        for layer_idx in range(self.num_layers):
            new_streams = []
            
            # Cross-attention: each stream attends to all streams
            for stream_idx, stream in enumerate(streams):
                # Concatenate all streams as key and value
                all_streams = torch.cat(streams, dim=1)  # [batch, num_streams, 512]
                
                # Cross-attention
                attn_output, attn_weights = self.fusion_layers[layer_idx][stream_idx](
                    query=stream,
                    key=all_streams,
                    value=all_streams
                )
                
                # Store attention weights for visualization
                if layer_idx == 0:
                    attention_weights_list.append(attn_weights)
                
                # Layer normalization + residual
                attn_output = self.layer_norms[layer_idx][stream_idx](
                    stream + attn_output
                )
                
                # Feed-forward network + residual
                ffn_output = self.ffn_layers[layer_idx][stream_idx](attn_output)
                ffn_output = attn_output + ffn_output
                
                new_streams.append(ffn_output)
            
            streams = new_streams
        
        # Final fusion: concatenate all streams and project to single vector
        concatenated = torch.cat([s.squeeze(1) for s in streams], dim=1)  # [batch, 4*512]
        fused_features = self.final_fusion(concatenated)  # [batch, 512]
        
        attention_weights = {
            'fusion_attention': attention_weights_list
        }
        
        return fused_features, attention_weights


class FakeNewsDetectionModel(nn.Module):
    """
    Complete Fake News Detection Model.
    Combines feature extraction, hierarchical fusion, and classification.
    """
    
    def __init__(self, feature_dim=512, num_heads=8, num_fusion_layers=2, dropout=0.1):
        super().__init__()
        
        # Hierarchical Fusion
        self.fusion = HierarchicalFusion(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            dropout=dropout
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2)  # 2 classes: Real, Fake
        )
        
    def forward(self, features_dict):
        """
        Args:
            features_dict: Dictionary with 4 feature streams
        
        Returns:
            logits: [batch_size, 2] - Classification logits
            fused_features: [batch_size, 512] - Fused representation
            attention_weights: Attention visualization data
        """
        # Hierarchical fusion
        fused_features, attention_weights = self.fusion(features_dict)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, fused_features, attention_weights


# Test the module
if __name__ == "__main__":
    print("Testing Hierarchical Fusion Module...\n")
    
    # Create dummy features
    batch_size = 8
    feature_dim = 512
    
    features = {
        'text_pattern': torch.randn(batch_size, feature_dim),
        'text_semantic': torch.randn(batch_size, feature_dim),
        'image_pattern': torch.randn(batch_size, feature_dim),
        'image_semantic': torch.randn(batch_size, feature_dim),
    }
    
    # Create model
    model = FakeNewsDetectionModel(feature_dim=512, num_heads=8, num_fusion_layers=2)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Forward pass
    logits, fused, attn = model(features)
    
    print(f"\n✓ Logits shape: {logits.shape} (expected: [8, 2])")
    print(f"✓ Fused features shape: {fused.shape} (expected: [8, 512])")
    print(f"\n✅ Hierarchical Fusion Module works!")
