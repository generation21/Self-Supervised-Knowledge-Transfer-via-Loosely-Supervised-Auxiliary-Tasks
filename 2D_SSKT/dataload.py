import torch
import torchvision.datasets as datasets
from torchvision import transforms
import torch.utils.data as data

def dataLoadFunc(opt):
    # Data loading parameters
    use_cuda = torch.cuda.is_available()
    params = {'batch_size': opt.batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),normalize
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),normalize
    ])
    
    

    train_set = datasets.STL10(root='/raid/video_data/stl10', split='train', download=True, transform = train_transform)
    valid_set = datasets.STL10(root='/raid/video_data/stl10', split='test', download=True, transform = val_transform)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    return train_loader, valid_loader

class DatasetPlaces365(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, transform=None):
        "Initialization"
        self.data_path = data_path
        self.transform = transform
        self.folders = folders
    
    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)
    
    def read_images(self, datapath, folder, use_transform):  
        image_path = folder[0]

        image = Image.open(os.path.join(datapath, *image_path.split('/'))).convert('RGB')

        if use_transform is not None:
            image = use_transform(image)

        return image

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        image = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        label = torch.LongTensor([int(folder[1])])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return image, label
