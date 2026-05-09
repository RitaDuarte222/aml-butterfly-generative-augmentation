import torchvision.transforms as transforms

# O tamanho exigido pela BaselineCNN dos professores é 64x64
IMAGE_SIZE = 64

# Estes valores foram calculados no notebook 00_dataset_exploration
BUTTERFLY_MEAN = [0.490, 0.468, 0.380] # Exemplo, atualizar com os reais se necessário
BUTTERFLY_STD  = [0.222, 0.219, 0.227]

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