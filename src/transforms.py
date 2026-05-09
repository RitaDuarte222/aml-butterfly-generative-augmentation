import torchvision.transforms as transforms

# O tamanho exigido pela BaselineCNN dos professores é 64x64
IMAGE_SIZE = 64

# Estes valores foram calculados no notebook 00_dataset_exploration
BUTTERFLY_MEAN = [0.480, 0.466, 0.338]
BUTTERFLY_STD  = [0.216, 0.211, 0.204] 

def get_transforms():
    # Transformações para o treino (com alguma normalização mas SEM data augmentation gerativa)
    # A Baseline deve ser treinada apenas com as imagens originais, mas podemos incluir transformações básicas
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=BUTTERFLY_MEAN, std=BUTTERFLY_STD)
    ])

    # Transformações para a validação (apenas redimensionar e normalizar)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BUTTERFLY_MEAN, std=BUTTERFLY_STD)
    ])

    return train_transform, val_transform