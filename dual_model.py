import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_backbone(name: str, pretrained: bool):
    import torchvision.models as M

    if name == "convnext_tiny":
        weights = M.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = M.convnext_tiny(weights=weights)
        # convnext classifier: (LayerNorm2d, Linear). We want features before classifier.
        dim = m.classifier[2].in_features
        m.classifier = nn.Identity()
        return m, dim

    if name == "resnet50":
        weights = M.ResNet50_Weights.DEFAULT if pretrained else None
        m = M.resnet50(weights=weights)
        dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, dim

    raise ValueError(f"Unsupported backbone: {name}")


class DualFusionNet(nn.Module):
    """
    Dual input model:
      - branch A: tight crop
      - branch B: context crop
    Fusion: concat([f_tight, f_ctx]) -> projection -> classifier

    Returns:
      logits, z (normalized embedding for SupCon)
    """

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, emb_dim: int = 256):
        super().__init__()
        self.backbone_name = backbone
        self.encoder, feat_dim = _make_backbone(backbone, pretrained=pretrained)

        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, emb_dim),
        )
        self.cls = nn.Linear(emb_dim, num_classes)

    @staticmethod
    def _to_vector(feat: torch.Tensor) -> torch.Tensor:
        """
        Ensure backbone output is (B, C).
        ConvNeXt returns a spatial tensor when classifier is removed.
        """
        if feat.dim() == 4:  # (B,C,H,W)
            feat = feat.mean(dim=(2, 3))
        elif feat.dim() == 3:  # (B,C,L) or (B,L,C) - fallback to mean over last dim if needed
            # pick a reasonable reduction to (B,C)
            if feat.size(1) <= feat.size(2):
                feat = feat.mean(dim=2)
            else:
                feat = feat.mean(dim=1)
        elif feat.dim() == 2:
            return feat
        else:
            feat = feat.reshape(feat.size(0), -1)
        return feat

    def forward(self, x_tight: torch.Tensor, x_ctx: torch.Tensor):
        f1 = self._to_vector(self.encoder(x_tight))
        f2 = self._to_vector(self.encoder(x_ctx))
        f = torch.cat([f1, f2], dim=1)
        z = self.proj(f)
        z_norm = F.normalize(z, dim=1)
        logits = self.cls(z)
        return logits, z_norm

