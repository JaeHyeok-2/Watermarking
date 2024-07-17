from watermarking_utils import Decoder, CustomDataset, get_data_loaders
import os 

import torch.nn as nn 
import torch 
import json 


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
    args = parser.parse_args()
    return args 

def validation(model, data_file_path, model_ckpt= None):
    _, valid_dataloader = get_data_loaders(file_path= data_file_path, batch_size=32) 

    metric_evaluation = {}
    message_length = model.message_length 

    pretrained_weight = f"SC-GS/watermarked_image/model_pth_message_length_{message_length}"
    if model_ckpt is not None: 
        model.load_state_dict(torch.load(os.path.join(pretrained_weight, f"watermarked_decoded_Epoch_{model_ckpt}.pth"))) 

    device = 'cuda' if torch.cuda.is.available() else 'cpu' 

    model.to(device) 
    model.eval() 

    val_dataloader = get_data_loaders(file_path, batch_size = 32, message_length = model.message_length, is_train=False) 
    
    total_bitwise_error = 0
    total_samples = 0

    with torch.no_grad():
        for idx, (image, message) in enumerate(valid_dataloader):
            val_image = image.to(device) 
            val_message = message.unsqueeze(0).repeat(batch_size, 1).to(device) 

            val_retrieval_messages = decoder(val_image) 
            val_loss += nn.MSELoss(val_retrieval_messages, val_message).item() 

            bitwise_avg_err = torch.sum(torch.abs(val_retrieval_messages_rounded - val_message.detach().cpu().numpy())) / (batch_size * messages.shape[-1])
            total_bitwise_error += bitwise_avg_err

        val_loss /= len(val_dataloader) 
        bitwise_avg_error = total_bitwise_error / len(valid_dataloader)

    print("Average Validation Loss : ", val_loss) 
    print("Bitwise average Error   : ", bitwise_avg_err)

    metric_evaluation['Val_Loss'] = val_loss 
    metric_evaluation['Bit_Avg_err'] = bitwise_avg_err

    make_metric_json(metric_evaluation)

