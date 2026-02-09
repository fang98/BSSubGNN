import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    use_uva = False
else:
    use_uva = True


const_args = {
    'device':device,
    'use_uva':use_uva
    }