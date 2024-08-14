import torch
import os


def test_torch():
    # Vérifie si PyTorch est installé
    print(f"PyTorch version: {torch.__version__}")

    # Crée un tenseur simple
    x = torch.rand(5, 3)
    print("Tensor x:\n", x)

    # Effectue une opération simple sur le tenseur
    y = torch.rand(5, 3)
    print("Tensor y:\n", y)
    z = x + y
    print("Result of x + y:\n", z)

    # Vérifie si le GPU est disponible et utilise-le
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")
        x = x.to(device)
        y = y.to(device)
        z = x + y
        print("Result of x + y on GPU:\n", z)
    else:
        print("CUDA is not available. Running on CPU.")


if __name__ == "__main__":
    print(os.environ.get('CONDA_PREFIX'))
    test_torch()
