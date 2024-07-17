# SC-GS Rendering 기반 Watermarking

## Task
- 기존 StegaNeRF의 `U-Net기반의 Decoder` + `Guided Classifier` $F$를 이용해서 Image내의 Message를 추출해보자 
- 사용된 FrameWork
    - [Invisible-Watermarking](https://github.com/ShieldMnt/invisible-watermark) : Image내에 bit, bytes, character를 DWTDCT 외 다른 기법들로 Watermarking 해주는 FrameWork
        - 해당 방식으로 Watermarking Data 생성


    - [StegaNeRF's Decoder & Classifier](https://github.com/XGGNet/StegaNeRF/blob/main/opt/unet.py#L17) : VGG16 기반의 Classifieir 와 U-Net기반의 Decoder에서 마지막 layer에 Message 길이를 추출하도록 Fc Layer를 추가 

    - [SC-GS](https://github.com/yihua7/SC-GS) : Rendering을 진행할 때 사용한 3DGS Model


    - [D-NeRF](https://github.com/albertpumarola/D-NeRF) :Dataest으로, mutant, lego, trex Dataset을 우선적으로 training 
# MileStone

### 1. Training Data, 즉 watermarking을 씌운 D-NeRF을 생성 
    - 이때, D-NeRF Dataset에는 다양한 Object에 대한 Image들이 있지만 우선적으로 Trex, Mutant, Lego 이 3개의 data만을 가지고 실행함
    - 각각의 Object에 대한 Dataset에는 다음과 같은 폴더로 구성되어있다.
    - Trex
        - train
            - r_0.png
            - r_1.png 
            - ...     
        - val
            - r_0.png
            - r_1.png
            - ...
        - test
            - r_0.png
            - r_1.png
            - ...


- 위와 같이 구성되어 있는 데이터셋에 대해서 **Invisible-Watermarking** 을 이용하여 Watermarking하여 non_watermarked_image 폴더에 저장
- **Watermarking Message 길이에 따라서 따로따로 데이터셋을 만들어야함. (Message=16 일때 데이터셋과 Message=32일때 Dataset을 따로)**



### 2. Message Decoder와 Classifier를 Training 
    - 1번에서 만든 워터마킹 데이터를 기반으로 Decoder와 Classifier를 학습을 진행
    - Message의 길이는 16과 32비트 둘 다 진행해봄
    

- **Classifier : VGG16기반의 Network로 입력으로 워터마킹 이미지를 받는다. 이후에 해당 Image x에 대해서 Watermarking의 확률 x_c를 반환, 단순히 디코더에서 워터마킹 메세지를 더 정확하게 뽑아내기 위한 가이드 역할**
- **Decoder: U-Net기반으로, 입력으로 워터마킹 이미지와 Classifier의 결과인 x_c를 함께 넣어서 Message를 복원하도록 설계, 기존 Stega-NeRF의 모델의 마지막 layer에 Message길이를 출력하도록 FC-layer를 추가** 





### 3. 워터마킹이 없는 SC-GS의 render Image에 대해서 성능 평가
- 워터마킹을 씌우지 않은 D-NeRF 데이터셋에 pretrained SC-GS를 적용하여 랜더링 이미지를 뽑아냄
    - 이때 기존의 D-NeRF 모델로 rendering할때는 pretrained-model이 있었지만, SC-GS에 대해서는 pretrained 된 모델을 못찾아서 동일하게 80_000iteration Training후 rendering 진행

- 이 뽑아낸 랜더링 이미지들은 결과적으로 워터마킹이 없는 이미지이다. 따라서, 2번 Message Decoder에 입력으로 넣는다면 기존 Message를 복원하지 못해야함
- 예를들어 b'1010101010'를 워터마킹했던 이미지들에 대해서 같은 메세지를 복원하는 Decoder에게 워터마킹을 하지않은 Raw Image를 넣게 되면 000100100 와 같이 복원을 잘 못해야함
- Bitwise Accuracy를 기반으로 Render Image로부터 추출한 Message와 original Messasge 사이의 Bit Accuracy를 측정( 낮을수록 좋음) 



## Code
1. ```train_gui.py``` : SC-GS의 training code로 D-NeRF의 Dataset을 기반으로 훈련

2. ```render.py``` : SC-GS를 Training 후, Image를 Rendering (여기서 만들어진 data로 3번 성능평가로 사용)

3. ```utils/watermarking.utils.py``` : Watermarking 필요한 utils 저장 

4. ```utils/stega_classifier.py``` : Stega-NeRF의 Decoder, Classifier 수정 

5. ```message_train.py``` : , Watermarking Image, Message를 기반으로 Decoder Training 

6. ```make_watermarking.py``` : Raw Image에 대해서 Invisible-Watermarking Framework를 통해서 watermarking 이미지 생성



## Notices
- 일단, D-NeRF의 이미지셋은 800x800x3의 크기를 가지고 있으나, SC-GS의 Rendering Image는 400x400x3의 Shape을 가지고 있기 때문에, 
- 기존 이미지셋에 Watermarking을 씌우기 위해서 400x400 Resize를 진행한 후, Watermarking 진행
- C


