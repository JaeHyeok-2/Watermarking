"""
1. Original Image(Ground Truth) + Watermarking Bits 
"""
import os 
import cv2 
import matplotlib 
import torchvision 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
import torch.optim as optim
import numpy as np 
"""
train.gui.py line 1087~ line 1105 
"""
from torchvision import transforms 
from imwatermark import WatermarkEncoder, WatermarkDecoder
import cv2
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from PIL import Image




def tensor_to_numpy(tensor_image):
    np_image = tensor_image.cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    np_image = (np_image * 255).astype(np.uint8)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return np_image


def numpy_to_tensor(image):
    tensor_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = np.transpose(tensor_image, (2, 0 ,1))
    tensor_image = torch.tensor(tensor_image, dtype=torch.float32) / 255.0

    return tensor_image


def embedding_Watermark(image, message):
    np_image = tensor_to_numpy(image)
    encoder = WatermarkEncoder()


    watermarking = message
    encoder.set_watermark('bits', watermarking)
    bgr_encoded = encoder.encode(np_image, 'dwtDct')

    tensor_encoded = numpy_to_tensor(bgr_encoded)
    return bgr_encoded, tensor_encoded , watermarking


def decoding_Watermark(image):
    np_image = tensor_to_numpy(image)
    decoder = WatermarkDecoder('bits', 32)
    watermark = decoder.decode(np_image,'dwtDct')
    return watermark


def save_tensor_image(watermarked_image, rendered_image, idx, data_name, watermarked):
    transform = transforms.ToPILImage()
    wm_image = transform(watermarked_image)
    rd_image = transform(rendered_image) 
    
    if watermarked:
        wm_image.save(f"watermarked_image/{data_name}/gt/{idx}.png")
        rd_image.save(f'watermarked_image/{data_name}/render/{idx}.png')
    

    else: 
        wm_image.save(f"non_watermarked_image/{data_name}/gt/{idx}.png")
        rd_image.save(f"non_watermarked_image/{data_name}/render/{idx}.png")


class ConvBNReLU(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNReLU, self).__init__() 

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)






##############
#HiDDeN Decoder
##############
class Decoder(nn.Module):
    """
    message_length = 16
    decoder_blocks=7 ,decoder_channels=64
    
    """


    def __init__(self,message_length=7, decoder_blocks=7, decoder_channels=64): 
        super(Decoder, self).__init__() 
        self.channels = decoder_channels 
        self.decoder_blocks = decoder_blocks
        self.message_length = message_length


        layers = [ConvBNReLU(3, self.channels)] 
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNReLU(self.channels, self.channels)) 

        layers.append(ConvBNReLU(self.channels, self.message_length))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))

        self.layers = nn.Sequential(*layers) 

        self.linear = nn.Linear(self.message_length, self.message_length) 

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)

        x.squeeze_(3),x.squeeze_(2)
        x = self.linear(x) 
        return x 



class CustomWatermarkDataset(Dataset):
    def __init__(self, non_watermarked_dir, watermarked_dir, message_length=16, transform=None,  watermarked_data=True):
        self.non_watermarked_dir = non_watermarked_dir 
        self.watermarked_dir = watermarked_dir 
        self.transform = transform 
        self.message_length = message_length 
        self.watermarked = watermarked_data

        self.images = []
        self.labels = []

        if self.watermarked :
            for img_name in os.listdir(watermarked_dir):
                self.images.append(os.path.join(watermarked_dir, img_name))
                label = torch.tensor([1 if i % 2 == 0 else 0 for i in range(message_length)], dtype=torch.float32)
                self.labels.append(label)
            
        else : 
            for img_name in os.listdir(non-watermarked_dir):
                self.images.append(os.path.join(non-watermarked_dir, img_name)) 
                self.labels.append(torch.zeros(message_length, dtype=torch.float32))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label





def get_data_loaders(file_path, batch_size=64, message_length=16, is_train=True, watermarked=True):
    # Get Watermarked Image & Rendered Image 


    data_transforms = {
        'train' : transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    }
    # train_dataset = CustomDataset(os.path.join(file_path,'gt'), transform = data_transforms['train'])
    # val_dataset = CustomDataset(os.path.join(file_path, 'render'), transform= data_transforms['test'])

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=False) 

    dataset_ = CustomWatermarkDataset(file_path[0], file_path[1], message_length, data_transforms['train'], watermarked_data=watermarked)
    
    if is_train: 
        dataset_length = len(dataset_) 
        train_length = int(0.9 * dataset_length) 
        val_length = dataset_length - train_length 
        train_subset, val_subset = random_split(dataset_, [train_length, val_length])
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


    dataloader_ = torch.utils.data.DataLoader(dataset_, batch_size=batch_size, shuffle=False)

    return dataloader_


def make_metric_json(metric_dict, save_path, file_name):

    with open(filename, 'w') as json_file:
        json.dump(metric_dict, json_file, indent=4) 

    print("Completed Saving Metric JSON FORMAT!!") 




class WatermarkExtractor(nn.Module):
    def __init__(self, lr = 1e-3):
        super(WatermarkExtractor, self).__init__() 
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.fc1 = nn.Linear(32 * 400 * 400, 128)
        self.fc2 = nn.Linear(128, 16)
        
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.decoder_loss = 1e9

    def forward(self, x):
        # print("Input Shape : " ,x.shape)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1,32 *400 * 400)
        # print("Flatten Image Shape : ", x.shape) # [32, 160000]
        x = self.relu(self.fc1(x)) 
        x = torch.sigmoid(self.fc2(x))

        return x 

    def save_weights(self, model_path, num_of_iteration):
        
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        file_path = f"{model_path}_{num_of_iteration}.pth"
        torch.save(self.state_dict(), file_path)


# summary(WatermarkExtractor().cuda(), (3,400, 400))

'''

def embed_message(image, message, alpha=0.9):
    watermarked_image = image.clone().float()
    message = [int(bit) for bit in message] 
    message_length = len(message)

    # print("Watermarked_image Shape : ", watermarked_image.shape) 
    # print("Message LENGTH : ", message_length)
    block_size = 8 
    idx = 0

    for channel in range(image.shape[0]):  # 각 채널에 대해 처리
        for i in range(0, image.shape[1], block_size):
            for j in range(0, image.shape[2], block_size):
                if idx >= message_length:
                    break
                block = image[channel, i:i+block_size, j:j+block_size].float().cpu().numpy()  # 각 채널의 블록 선택 후 NumPy 배열로 변환
                dct_block = cv2.dct(block)
                if message[idx] == 1:
                    dct_block[4, 4] += alpha
                else:
                    dct_block[4, 4] -= alpha
                idx += 1
                idct_block = cv2.idct(dct_block)
                watermarked_image[channel, i:i+block_size, j:j+block_size] = torch.tensor(idct_block)

    return watermarked_image.clip(0, 255).to(image.device), message

def save_extracted_message_log(message):
    log_file = 'Extracted_log/extracted_message_iter150K_lambda_0_1.txt' 

    with open(log_file, 'a') as f:
        for i, val in enumerate(message):
            val_int = torch.round(val)
            f.write(f"Bit {i}: {val_int}\n")

'''

def bit_acc(decoded, original):
    diff = (~torch.logical_xor(decoded>0, original> 0))
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]

    return bit_accs 

