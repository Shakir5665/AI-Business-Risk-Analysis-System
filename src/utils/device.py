import torch

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("="*50)
        print("GPU Available")
        print("="*50)
        print(f"Device : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version : {torch.version.cuda}")
        print(f"PyTorch : {torch.__version__}")
        print("="*50)

    else:

        device = torch.device("cpu")

        print("="*50)
        print("Running on CPU")
        print("="*50)

    return device