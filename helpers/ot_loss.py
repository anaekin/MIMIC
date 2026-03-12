import torch
import torch.nn as nn

from data.templates import get_templates

class ManifoldTransportLoss(nn.Module):
    def __init__(self, 
                 text_encoder, 
                 tokenizer, 
                 embedding_dim, 
                 target_text, 
                 device='cuda', 
                 n_projections=128):
        """
        Args:
            text_encoder: Function/Module that takes tokenized text and returns embeddings.
            tokenizer: Tokenizer corresponding to the text_encoder.
            embedding_dim: The dimension of the shared space (D).
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.n_projections = n_projections
        self.embedding_dim = embedding_dim
        
        # Pre-compute the Target Manifold (The "Platonic Cloud")
        self.target_dist = self.create_text_manifold(target_text)

    def create_text_manifold(self, class_name, n_samples=500):
        # 1. Expand prompts to create the statistical cloud
        prompts = get_templates(class_name, n_samples)
        
        # 2. Tokenize and Encode
        # ADAPTATION: Handle specific tokenizer call signatures here
        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            # Get embeddings. 
            # If LLaVA/LLM: This is usually model.get_input_embeddings()(tokens.input_ids)
            # If CLIP: This is model.encode_text(tokens)
            text_features = self.text_encoder(tokens)
            
            # Critical: Normalize if the space relies on Cosine Similarity (like CLIP)
            # Optional if using raw Euclidean space of an LLM
            if hasattr(text_features, 'norm'): 
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
        return text_features.detach() # Shape: (N_samples, Dim)

    def forward(self, image_embeddings):
        """
        Args:
             image_embeddings: (B, N_patches, Dim) 
             The output of your Vision Encoder (or Adapter) before the final logits.
        """
        # Flatten patches
        # print(f"emb dim:{self.embedding_dim} | image embeds: {image_embeddings.shape}")
        source_dist = image_embeddings.view(-1, self.embedding_dim)
        target_dist = self.target_dist
        
        # --- Sliced Wasserstein Distance (SWD) ---
        
        # 1. Random Projections
        projections = torch.randn(self.n_projections, self.embedding_dim).to(self.device)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)
        
        # 2. Project & Sort
        source_proj = source_dist @ projections.T
        target_proj = target_dist @ projections.T
        
        source_sorted, _ = torch.sort(source_proj, dim=0)
        
        # Resample target to match source size (if needed)
        if target_proj.shape[0] != source_proj.shape[0]:
            target_proj = self.resample_target(target_proj, source_proj.shape[0])
        
        target_sorted, _ = torch.sort(target_proj, dim=0)
        
        # 3. Transport Cost
        return torch.mean((source_sorted - target_sorted) ** 2)

    def resample_target(self, target, n_target):
        indices = torch.randint(0, target.shape[0], (n_target,), device=self.device)
        return target[indices]