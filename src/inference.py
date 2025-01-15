import torch
import matplotlib.pyplot as plt

def remove_bone_shadow(model, input_image: torch.Tensor, output_path: str, use_cmap: bool = False):
    input_image = input_image.to(next(model.parameters()).device)
    
    with torch.no_grad():
        output = model(input_image)
    
    output_np = output.squeeze().cpu().numpy()
    
    cmap = "ocean_r" if use_cmap else "gray"
    
    plt.figure(figsize=(6, 6))
    plt.imshow(output_np, cmap=cmap)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()