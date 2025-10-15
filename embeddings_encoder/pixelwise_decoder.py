import torch.nn as nn

class PixelwiseCompressor(nn.Module):
    """
    Simple MLP-based compressor that processes each pixel independently,
    compressing from 768 dimensions to bottleneck_dim and back.
    """
    def __init__(self, input_dim=768, bottleneck_dim=10):
        super(PixelwiseCompressor, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, bottleneck_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim // 2),
            nn.ReLU(inplace=True), 
            nn.Linear(input_dim // 2, input_dim),
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return compressed, reconstructed
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)