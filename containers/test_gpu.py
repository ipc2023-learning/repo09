import torch


def main():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available.")
        device = torch.device("cuda")
        print(f"GPU Device: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    main()
