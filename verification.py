import torch

# Verifica si PyTorch tiene soporte para CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA disponible?", device)

if device.type == 'cuda':
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria total de la GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2} MB")