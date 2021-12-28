# This is the config file
from models import *
import torchvision.transforms as transforms

####################################################
# about training
learning_rate = 0.01

train_batch_size = 128
test_batch_size = 128

epoch_num = 2

train_show_interval = 5

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

####################################################
# about persist
persist_dir_name = 'checkpoint'

resume = True
save_flag = True

img_path = "./img/"

####################################################
# Choose a net here
net = LeNet()
#net = VGG('VGG16')
