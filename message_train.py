import numpy as np
import os
import re
import csv
import time
import pickle
import logging
from tqdm import tqdm 

import torch 
import torch.nn as nn
import torch.optim as optim 
import argparse

from torchvision import datasets, transforms 
from utils.watermarking_utils import  CustomWatermarkDataset, get_data_loaders
from utils.stega_classifier import VGG16UNet, Classifier
from PIL import Image 
from torch.utils.data import Dataset, DataLoader,Subset






def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))

    return this_run_folder


def decoded_message(predicted_message_prob):
    print("Predicted Message  Score : ", predicted_message_prob) 
    print("Binarized Mssage :" , torch.round(predicted_message_prob)) 
    

def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    

    # Add arguments
    parser.add_argument('--pretrained', type=bool, default=False, help='Loading Pretrained Model?')
    parser.add_argument('--subset', type=bool, default=False, help='Use 1%_of the entire dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lambda_dec', type=float, default=0.8, help='Regularization term of Decoding')
    parser.add_argument('--lambda_cls', type=float, default=0.2, help='Regularization term of classification')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--data_name', type=str, default='lego', help="bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex")
    parser.add_argument('--message_length', type=int, default=16, help='Watermarking Message length')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate for Cls & Dec')
    parser.add_argument('--watermarked',type=bool, default=True, help='Whether Watermarked Data or not')
    parser.add_argument('--just_eval', type=bool, default=False, help='ONLY validation?')
    args = parser.parse_args()
    return args 

if __name__ =="__main__":


    args = parse_args() 
    print(f"Batch size:              {args.batch_size}")
    print(f"Lambda_dec:              {args.lambda_dec}")
    print(f"Lambda_cls:              {args.lambda_cls}")
    print(f"Epochs:                   {args.epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Device:                  {args.device}")
    print(f"Data name:               {args.data_name}")
    print(f"Message_length :         {args.message_length}")

    if not os.path.exists(f'watermarked_decoder_pth/{args.data_name}'):
        os.makedirs(f'watermarked_decoder_pth/{args.data_name}', exist_ok=True)

    if not os.path.exists(f'watermarked_decoder_classification_pth/{args.data_name}'):
        os.makedirs(f'watermarked_decoder_classification_pth/{args.data_name}', exist_ok=True)


    # training, validation = [watermarked_image/gt, watermarked_image/render]

    print_epoch = 10
    Message_Classifier = Classifier()   
    decoder = VGG16UNet(message_length=args.message_length)


    if args.pretrained: 
        dec_model_path = f"watermarked_decoder_pth/{args.data_name}/model_pth_message_length_{args.message_length}"
        cls_model_path = f"watermarked_decoder_classification_pth/{args.data_name}/model_pth_message_length_{args.message_length}"
        files = os.listdir(dec_model_path) 
        
        idx = None
        for file in files:
            parts = file.split('_') 
            parts = parts[-1]
            epoch_idx, pth = int(parts.split('.')[0]), parts.split('.')[1]
            if idx is None or epoch_idx > idx : 
                idx = epoch_idx

        idx = 4
        print(os.path.join(dec_model_path,f"watermarked_decode_Epoch_{idx}.pth"))
        decoder.load_state_dict(torch.load(os.path.join(dec_model_path,f"watermarked_decode_Epoch_{idx}.pth")))
        Message_Classifier.load_state_dict(torch.load(os.path.join(cls_model_path,f"watermarked_decode_Epoch_{idx}.pth")))
        print("Successfully Loading Decoder & Message_classifier!! EPOCH :{}".format(idx))

    
    # decoder = Decoder(message_length=16, decoder_blocks=7, decoder_channels=64) # HiDDeN 
    
    file_path = [os.path.join('non_watermarked_image',f"{args.data_name}"), os.path.join('watermarked_image_32',f"{args.data_name}")]

    if not args.just_eval: 
        """StegaNeRF's Encoder (UNet) --> Classifier + VGG16UNet base"""



        # print(decoder.message_length)
        train_dataloader, val_dataloader = get_data_loaders(file_path, batch_size=args.batch_size, message_length = args.message_length, is_train=True, watermarked=args.watermarked)
        # val_dataloader = get_data_loaders(file_path, batch_size=args.batch_size, message_length = args.message_length, watermarked=args.watermarked)

        file_train_count = len(train_dataloader.dataset)
        file_val_count = len(val_dataloader.dataset)

        print("Number of training Images   : ", file_train_count)
        print("Number of validation Images : ", file_val_count)

        bce_with_logits_loss = nn.BCEWithLogitsLoss().to(args.device)

        mse_loss = nn.MSELoss().to(args.device)
        ce_loss = nn.BCEWithLogitsLoss().to(args.device)

        optimizer = torch.optim.Adam(list(decoder.parameters()) + list(Message_Classifier.parameters()), lr= args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        


        decoder.to(args.device)
        Message_Classifier.to(args.device)


        best_loss =1e9
        message_length = args.message_length

        if args.subset:
            dataset_size = len(train_dataloader.dataset) 
            indices = list(range(dataset_size))
            np.random.shuffle(indices)

            subset_size = int(dataset_size * 0.01)
            subset_indices = indices[:subset_size] 

            train_subset = Subset(train_dataloader.dataset, subset_indices)
            val_subset = Subset(val_dataloader.dataset, subset_indices) 

            train_dataloader, val_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True), DataLoader(val_subset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        


        for epoch in tqdm(range(args.epochs)):

            epoch_start = time.time() 
            decoder.train() 

            total_matches = 0
            total_bits = 0
            for idx, (image,message) in enumerate(train_dataloader):

                image = image.to(args.device) 
                # messages = message.unsqueeze(0).repeat(batch_size,1).to(device)
                messages = message.to(args.device)

                classification_label = torch.ones((image.size(0), 1), dtype=torch.float32).to(args.device)
                predicted_x_c = Message_Classifier(image)
                
                retrieval_messages = decoder(image, predicted_x_c) 
                # print(f'Retrieval_message : {retrieval_messages.dtype}, Original_message : {messages.dtype}')
                loss_dec = mse_loss(retrieval_messages, messages).to(args.device)
                loss_cls = ce_loss(predicted_x_c, classification_label)
                

                # print("Original Message   : ", messages)
                # print("Retrieval Messsage : ", torch.round(retrieval_messages))
                # print(loss_dec, loss_cls)

                loss = args.lambda_dec * loss_dec + args.lambda_cls * loss_cls 
                # loss_dec.backward()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()            

                retrieval_messages_rounded = retrieval_messages.detach().cpu().round().clip(0, 1)

                bitwise_matches = (message == retrieval_messages_rounded).sum() 
                total_matches += bitwise_matches 

                total_bits += retrieval_messages.numel()
            print("Bitwise Accuracy :", (total_matches / total_bits).item() *100,'%')       
                        
            if epoch % 10 == 0 : 
                print(f"{epoch+1}EPOCH, {idx+1}th Iteration's Loss : {loss_dec}")
                # print(retrieval_messages_rounded.shape, messages.shape)


            scheduler.step()

            train_duration = time.time() - epoch_start 


            # print(f" 1 EPOCH TRAINING TIME : {train_duration}")

            decoder.eval()

            val_loss = 0.0 

            with torch.no_grad():
                val_total_matches = 0
                val_total_bits = 0
                for idx, (val_image, val_message) in enumerate(val_dataloader):
                    val_image = val_image.to(args.device) 
                    val_messages = val_message.to(args.device)
                    
                    classification_label = torch.ones((image.size(0), 1), dtype=torch.float32).to(args.device)
                    predicted_x_c = Message_Classifier(val_image)
                    
                    val_retrieval_messages = decoder(val_image, predicted_x_c) 
                    val_loss += mse_loss(val_retrieval_messages, val_messages).item()


                    val_retrieval_messages_rounded = val_retrieval_messages.detach().cpu().round().clip(0, 1)
                    val_bitwise_matches = (val_messages.cpu() == val_retrieval_messages_rounded).sum() 
                    val_total_matches += val_bitwise_matches 

                    val_total_bits += val_retrieval_messages.numel()
            
                    # print(val_loss)
            val_loss /= len(val_dataloader) 
                    
            print(f"Epoch {epoch+1}, Validation Loss : {val_loss}")
            print("Bitwise Accuracy :", (total_matches / total_bits).item() *100,'%')   
            
            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping_counter = 0 
                if not os.path.exists(f'watermarked_decoder_pth/{args.data_name}/model_pth_message_length_{args.message_length}'):
                    os.makedirs(f'watermarked_decoder_pth/{args.data_name}/model_pth_message_length_{args.message_length}', exist_ok=True)
                torch.save(decoder.state_dict(),f'watermarked_decoder_pth/{args.data_name}/model_pth_message_length_{args.message_length}/watermarked_decode_Epoch_{epoch}.pth')

                if not os.path.exists(f'watermarked_decoder_classification_pth/{args.data_name}/model_pth_message_length_{args.message_length}'):
                    os.makedirs(f'watermarked_decoder_classification_pth/{args.data_name}/model_pth_message_length_{args.message_length}', exist_ok=True)
                torch.save(Message_Classifier.state_dict(),f'watermarked_decoder_classification_pth/{args.data_name}/model_pth_message_length_{args.message_length}/watermarked_decode_Epoch_{epoch}.pth')

                print(f"Saving {epoch+1} EPOCH WEIGHTS!")
            else:
                early_stopping_counter +=1 
            

            if early_stopping_counter >= args.early_stopping_patience:
                print("Early Stopping Triggered")
                break 


    # Just Evaluation for non-watermarking data
    else: 

        train_dataloader = get_data_loaders(file_path, batch_size=args.batch_size, message_length = args.message_length, is_train=False, watermarked=args.watermarked)
        print(f"Number of Non-Watermarking Dataset : {len(train_dataloader.dataset)} ")

        decoder.eval()
        Message_Classifier.eval()

        decoder.to(args.device)
        Message_Classifier.to(args.device)

        total_matches = 0
        total_bits = 0
 
        with torch.no_grad():
            for idx, (val_image, _) in enumerate(train_dataloader):
                current_batch_size = len(val_image)
                val_image = val_image.to(args.device) 

                predicted_x_c = Message_Classifier(val_image)
                   
                val_retrieval_messages = decoder(val_image, predicted_x_c) 
                retrieval_messages_rounded = val_retrieval_messages.detach().cpu().round().clip(0, 1)
                
                watermarking_message = torch.tensor([1 if i % 2 == 0 else 0 for i in range(args.message_length)], dtype=torch.float32).repeat(current_batch_size, 1)
                
                # print(retrieval_messages_rounded.shape, watermarking_message.shape)
                bitwise_matches = (watermarking_message == retrieval_messages_rounded).sum() 
                total_matches += bitwise_matches 
                total_bits += watermarking_message.numel()

        print("Bitwise Accuracy :", (total_matches / total_bits).item() *100,'%')                




    torch.cuda.empty_cache()


