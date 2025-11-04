import torch
import torch.nn as nn
import timm

class HybridModel(nn.Module):
    def __init__(self, num_classes=2, effnet_name='efficientnet_b0',
                 vit_name='vit_tiny_patch16_224', in_chans=1,
                 use_vit=True, freeze_backbone=False):
        super().__init__()
        self.in_chans = in_chans
        self.use_vit = use_vit

        # CNN Backbone
        self.cnn = timm.create_model(
            effnet_name,
            pretrained=True,
            in_chans=in_chans,
            num_classes=0  # No final classification layer
        )
        cnn_feat_dim = self.cnn.num_features  # 1280 for efficientnet_b0

        # ViT Backbone
        if use_vit:
            self.vit = timm.create_model(
                vit_name,
                pretrained=True,
                in_chans=3,
                num_classes=0
            )
            vit_feat_dim = self.vit.num_features  # 192 for vit_tiny
            
            # Project CNN features to ViT dimension
            self.cnn_proj = nn.Linear(cnn_feat_dim, vit_feat_dim)
            
            # Cross-attention layer
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=vit_feat_dim,
                num_heads=4,
                dropout=0.1
            )
        else:
            self.vit = None
            vit_feat_dim = 0

        # Classifier Head
        total_feat = cnn_feat_dim + (vit_feat_dim if use_vit else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_feat),
            nn.Linear(total_feat, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        # Input verification
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        
        # CNN pathway
        cnn_feat = self.cnn(x)  # [B, 1280]
        
        # ViT pathway
        if self.use_vit:
            vit_x = x.repeat(1,3,1,1)  # 1->3 channels
            vit_feat = self.vit(vit_x)  # [B, 192]
            
            # Project CNN features
            cnn_proj = self.cnn_proj(cnn_feat)  # [B, 1280] -> [B, 192]
            
            # Cross-attention (query=CNN, key/value=ViT)
            attn_out, _ = self.cross_attn(
                query=cnn_proj.unsqueeze(0),  # [1, B, 192]
                key=vit_feat.unsqueeze(0),    # [1, B, 192]
                value=vit_feat.unsqueeze(0)   # [1, B, 192]
            )
            
            # Combine features
            fused_feat = torch.cat([cnn_feat, attn_out.squeeze(0)], dim=1)  # [B, 1280+192]
        else:
            fused_feat = cnn_feat
            
        return self.classifier(fused_feat)

    def freeze_backbone(self):
        for p in self.cnn.parameters():
            p.requires_grad = False
        if self.vit is not None:
            for p in self.vit.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.cnn.parameters():
            p.requires_grad = True
        if self.vit is not None:
            for p in self.vit.parameters():
                p.requires_grad = True