import torch

if torch.cuda.is_available():
    print("GPUs are available!")
    
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs:", num_gpus)
else:
    print("No GPUs found.")

device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())]    
print(device_list[0])