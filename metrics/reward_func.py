import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import os

class AestheticMLP(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class SemanticAestheticReward(nn.Module):
    def __init__(self, 
                 device='cuda', 
                 clip_model="ViT-L/14", 
                 alpha=1.0,  # Weight for Semantic Score
                 beta=0.2    # Weight for Aesthetic Score
                 ):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        
        print(f"Loading CLIP {clip_model}...")
        
        # Check for local clip model first to avoid internet access
        # Mapping "ViT-L/14" to a local filename convention if needed, or check download_root
        local_clip_path = "./metrics/ckpt/ViT-L-14.pt"
        if os.path.exists(local_clip_path):
             print(f"Loading local CLIP weights from {local_clip_path}")
             self.clip_model, self.preprocess = clip.load(local_clip_path, device=device, jit=False)
        else:
             # Try standard load but warn/hope it's cached. 
             # To ensure NO internet, we should probably fail if not cached, but clip.load checks cache.
             # We can't easily patch clip.load without monkey patching.
             # But we can assume the user has placed it in ./ckpt/ if they are strict.
             # Or we look in ~/.cache/clip
             print("Attempting to load CLIP from default cache (or download if allowed/possible)...")
             self.clip_model, self.preprocess = clip.load(clip_model, device=device, jit=False)
             
        self.clip_model.eval()
        
        # --- Load LAION-Aesthetics V2 Head ---
        # The V2 predictor is a simple Linear Layer (768 -> 1) for ViT-L/14
        # Note: If using ViT-B/32, input dim is 512. V2 is specifically trained on L/14.
        self.aesthetic_head = AestheticMLP(768)
        self.load_aesthetic_weights()
        self.aesthetic_head.to(device)
        self.aesthetic_head.eval()

    def load_aesthetic_weights(self):
        """Downloads and loads the official LAION-Aesthetics V2 weights."""
        # URL for the standard V2 predictor (L/14)
        # weights_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac+logos+ava1-l14-linearMSE.pth?raw=true"
        weights_path = "./metrics/ckpt/sac_logos_ava1-l14-linearMSE.pth"
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Aesthetic weights not found at {weights_path}. Please download them manually to avoid internet access.")
            # print(f"Downloading Aesthetic Predictor V2 weights to {weights_path}...")
            # urllib.request.urlretrieve(weights_url, weights_path)
            
        state_dict = torch.load(weights_path, map_location="cpu")
        self.aesthetic_head.load_state_dict(state_dict)
        print("Aesthetic weights loaded.")

    def forward(self, images, text_prompts):
        """
        Args:
            images: Tensor of shape (B, C, H, W). Assumes normalized to CLIP standards.
                   If not normalized, apply self.preprocess manually before passing.
            text_prompts: List of strings (length B).
        Returns:
            total_reward: Tensor (B,)
            metrics: Dict containing 'semantic_score' and 'aesthetic_score'
        """
        batch_size = images.shape[0]
        
        # 1. Tokenize Text
        text_tokens = clip.tokenize(text_prompts, truncate=True).to(self.device)
        
        with torch.no_grad():
            # 2. Compute Embeddings (Shared Backbone)
            # Resize images to match CLIP model's expected input resolution
            target_resolution = self.clip_model.visual.input_resolution
            if images.shape[-1] != target_resolution or images.shape[-2] != target_resolution:
                 images = F.interpolate(images, size=(target_resolution, target_resolution), mode='bicubic', align_corners=False)

            # We run the heavy ViT only ONCE
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(text_tokens)

            # 3. Normalize Embeddings (Critical for Cosine Sim & Aesthetic MLP)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 4. Semantic Score (Cosine Similarity)
            # shape: (B, 1)
            semantic_score = (image_features * text_features).sum(dim=1, keepdim=True)
            
            # 5. Aesthetic Score (MLP Prediction)
            # The predictor outputs a value roughly between 1-10
            aesthetic_score = self.aesthetic_head(image_features.float())
            
            # Normalize aesthetic score to usually be in 0-1 range for stability in RL
            # LAION scores are 1-10. We scale it roughly to 0-1.
            aesthetic_norm = (aesthetic_score - 4.5) / 5.0 

            # 6. Combine
            total_reward = (self.alpha * semantic_score) + (self.beta * aesthetic_norm)

        return total_reward.squeeze(), {
            "semantic_score": semantic_score.squeeze(),
            "aesthetic_score": aesthetic_score.squeeze()
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Mock Data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_model = SemanticAestheticReward(device=device)
    
    # Random tensor simulating a batch of generated images (3x224x224)
    # NOTE: Ensure you apply CLIP normalization: (mean=[0.481, 0.457, 0.406], std=[0.268, 0.261, 0.275])
    mock_images = torch.randn(4, 3, 224, 224).to(device) 
    prompts = ["a photo of a goldfish", "a photo of a spider", "a red car", "abstract noise"]
    
    rewards, metrics = reward_model(mock_images, prompts)
    
    print("\n--- Results ---")
    for i, p in enumerate(prompts):
        print(f"Prompt: '{p}'")
        print(f"  Semantic: {metrics['semantic_score'][i]:.4f}")
        print(f"  Aesthetic (1-10): {metrics['aesthetic_score'][i]:.4f}")
        print(f"  Total Reward: {rewards[i]:.4f}")