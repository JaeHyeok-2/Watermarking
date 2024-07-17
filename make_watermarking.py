import os
import torch
from torchvision import transforms
from PIL import Image
from utils.watermarking_utils import embedding_Watermark, numpy_to_tensor, tensor_to_numpy, decoding_Watermark
# 이미지 변환 정의


data_transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
])

def process_and_save_images(src, dst):
    global idx
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    for root, _, files in os.walk(src):
        for file in files:
            src_file = os.path.join(root, file)
            image = Image.open(src_file).convert("RGB")
            tensor_image = data_transform(image)
            bgr_encoded, tensor_encoded, watermarking = embedding_Watermark(tensor_image, b'10101010101010101010101010101010')
            
            tensor_encoded = transforms.ToPILImage()(tensor_encoded) 
            file = str(idx) + file
            png_dst_file = os.path.join(dst, file)
            # pil_image = Image.fromarrays(tensor_encoded)
            pil_image = tensor_encoded
            pil_image.save(png_dst_file)
            print(png_dst_file)
            transform = transforms.ToTensor()
            tensor_encoded = transform(tensor_encoded)
            watermarking = decoding_Watermark(tensor_encoded)
            # print(watermarking)
            idx +=1

            # print(f"Saved PNG for {src_file} to {png_dst_file}")

def process_dataset(dataset_name):
    global idx 
    base_src_path = os.path.join('data', dataset_name)
    base_dst_path = os.path.join('watermarked_image_32', dataset_name)
    
    subsets = ['test', 'train', 'val']
    for subset in subsets:
        src_path = os.path.join(base_src_path, subset)
        process_and_save_images(src_path, base_dst_path)

if __name__ == "__main__":
    idx = 0
    datasets = ['lego', 'trex', 'mutant']

    for dataset in datasets:
        process_dataset(dataset)