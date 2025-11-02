import torch
from models.humansegment_unet import Segment

def create_segment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Segment] Using device: {device}")
    return Segment(max_size=512, outline_thickness=1, device=device)
