import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM)
    Focuses on 'what' features are important across channels.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (part of CBAM)
    Focuses on 'where' important features are located in spatial dimensions.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class HybridTBPredictor(nn.Module):
    """
    Hybrid CNN + Attention Model for Tuberculosis Detection
    Uses pre-trained DenseNet121 as feature extractor backbone
    coupled with Channel and Spatial Attention Blocks (CBAM).
    """
    def __init__(self, pretrained=True):
        super(HybridTBPredictor, self).__init__()
        
        # Load pre-trained DenseNet121 backbone
        if pretrained:
            densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            densenet = models.densenet121()
            
        # Extract features (before the final classifier)
        # DenseNet121 features output has 1024 channels
        self.features = densenet.features
        self.num_features = 1024
        
        # Attention Blocks
        self.channel_attention = ChannelAttention(self.num_features)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Adaptive pooling to reduce spatial dimension to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1) # Single output for Binary Classification (TB vs Normal)
        )

    def forward(self, x):
        # 1. Feature Extraction via DenseNet backbone
        # Output shape: [B, 1024, H_feat, W_feat]
        features = self.features(x)
        
        # 2. Apply Attention blocks (CBAM)
        # Channel Attention
        ca_out = self.channel_attention(features)
        feat_ca = features * ca_out
        
        # Spatial Attention
        sa_out = self.spatial_attention(feat_ca)
        feat_sa = feat_ca * sa_out
        
        # 3. Pooling and Flattening
        pooled = self.avgpool(feat_sa)
        flat = torch.flatten(pooled, 1)
        
        # 4. Classification
        logits = self.classifier(flat)
        return logits

if __name__ == "__main__":
    # Test instantiation and forward pass
    model = HybridTBPredictor(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)
    print("Model Input Shape:  ", dummy_input.shape)
    print("Model Output Shape: ", outputs.shape)
    print("Model initialized and verified successfully!")
