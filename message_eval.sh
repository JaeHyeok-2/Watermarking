# python message_train.py --pretrained=True --data_name=trex --watermarked=False --just_eval=True --message_length=32 #  2~3epoch
# python message_train.py --pretrained=True --data_name=mutant --watermarked=False --just_eval=True --message_length=32 # 3epoch
python message_train.py --pretrained=True --data_name=lego --watermarked=False --just_eval=True --message_length=32 # 4epoch

# Message Length = 16

    # 3epoch -> accuracy almost 100%

    # Mutant  pretrained idx = 0 -> bit accuracy = 81.25s
        # training data : 
    # Lego    pretrained idx = 0 -> bit accuracy = 99.73
    # trex    pretrained idx = 0 -> bit accuracy = 100

# Message Length = 32

    # Mutant  pretrained idx = 3 -> bit accuracy = 81.25    lambda_dec, cls = 0.5 , 0.3 epoch 3 -> 
    # Lego    pretrained idx = 4 -> bit accuracy = 99.73    
    # trex    pretrained idx = 4 -> bit accuracy = 100
    