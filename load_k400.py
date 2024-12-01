from torchvision.datasets import Kinetics

# Specify the root directory where you want to save the dataset
root_dir = "./kinetics_dataset"

# Parameters for the dataset
frames_per_clip = 16
num_classes = "400"  # Choose between '400', '600', or '700'
split = "train"  # Can be 'train', 'val', or 'test'
download = True  # Enable downloading

# Download and prepare the dataset
kinetics_dataset = Kinetics(
    root=root_dir,
    frames_per_clip=frames_per_clip,
    num_classes=num_classes,
    split=split,
    download=download
)
