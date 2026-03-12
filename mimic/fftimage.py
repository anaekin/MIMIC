import torch
import torch.nn as nn

class FFTImage(nn.Module):
    def __init__(self, batch_size=1, h=224, w=224, decay_power=1.0):
        super().__init__()
        self.h, self.w = h, w
        self.batch_size = batch_size
        # Initialize frequencies (Complex numbers)
        # Shape: (B, C, H, W // 2 + 1, 2)
        self.freqs = nn.Parameter(torch.randn(batch_size, 3, h, w // 2 + 1, 2) * 0.01)
        
        # Create a scaling mask that dampens high frequencies
        # The higher the frequency (closer to edges), the smaller the gradient update
        y = torch.linspace(-1, 1, h).view(-1, 1)
        x = torch.linspace(-1, 1, w // 2 + 1).view(1, -1)
        dist = torch.sqrt(x**2 + y**2) # Distance from center (DC component)
        scale = 1.0 / (dist**decay_power + 1e-4)
        # Shape: (1, 1, H, W // 2 + 1, 1) for broadcasting
        # We use register_buffer ensuring it's on the correct device when moved
        self.register_buffer("scale", scale.view(1, 1, h, w // 2 + 1, 1))

    def forward(self):
        # 1. Scale frequencies (Implicit Regularization)
        scaled_freqs = self.freqs * self.scale
        
        # 2. Convert Complex -> Real
        spectrum = torch.view_as_complex(scaled_freqs)
        
        # 3. Inverse FFT to get Image
        image = torch.fft.irfft2(spectrum, s=(self.h, self.w), norm='ortho')
        
        # 4. Sigmoid to constrain to [0,1]
        return torch.sigmoid(image)