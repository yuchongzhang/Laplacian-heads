"""Image-to-patch embedding. Copyright 2020 Ross Wightman."""
from torch import nn
from .helpers import to_2tuple
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, device=None, dtype=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype)
        self.norm = nn.Identity()
    def forward(self, x):
        assert x.shape[-2:] == self.img_size, f'Expected images of size {self.img_size}'
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))
