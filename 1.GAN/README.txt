main.py
GAN 모델을 트레이닝하고 트레이닝된 모델과 파라메터들을 저장하는 파이썬 스크립트입니다. 
main.py를 실행하기 위해서는 다양한 argument들을 설정해 주어야 합니다. 각 argument 들에 대한 설명은 아래와 같습니다.

--data_path: GAN을 트레이닝하기 위한 현대의 인풋 이미지 데이터가 저장되어 있는 디렉토리 경로입니다.

--datashape: original과 dropempty 중에 선택해야합니다. original은 들어온 이미지 높이 10 폭 30 RGB 3을 그대로 데이터로 가져온다는 뜻으로 보시면 되고, dropempty를 선택하시면 검은픽셀(정보가 없는)만 존재하는 컬럼들을 모두 빼고 트레이닝을 할것을 명명하는 것으로 이미지 높이 10 폭 14 RGB 3을 데이터로 가져온다는 뜻입니다.모델 중 self-attention을 사용한 모델로 트레이닝 할 때는 dropempty를 강력하게 추천드립니다 (메모리 이슈 관련). 디폴트값은 dropempty입니다.

--gan_model: GAN 모델 중 이번 R&D 태스크를 위하여 Info GAN 모델과 기본 모델 (vanila)을 사용하였습니다. 그래서 트레이닝을 할 GAN 모델을 infogan과 vanila 중에서 선택하셔야 합니다.

--loss_mode: GAN의 손실함수를 어떤 방법으로 계산할 것인지 선택하는 부분입니다. vanila는 기본 모델에서 사용하는 loss term 방식이고, wgan은 Wasserstein loss가 loss term으로 사용되는 것이고 wgangp는 gradient penalty부분이 향상된 wasserstein loss with gradient penalty가 적용이 된 loss term입니다. vanila, wgan 그리고 wgangp 중에서 선택해야합니다. 일반적으로 vanila보다 wgan이나 wgangp가 더 좋은 결과를 보이는 경향이 있는 것 같습니다. --gan_model을 vanila로 사용하는 경우, wgan이나 wgangp만 지원합니다.

--gen_mode: GAN의 generator에서 어떤 모델을 사용할 지 입니다. vanila는 MLP레이어로 구성되어있는 심플한 모델이고 selfattention은 self-attention mechanism을 도입한 모델입니다. vanila와 selfattention중에 선택하셔야 합니다. selfattention은 vanila에 비해 트레이닝 속도가 아주 느립니다. 그 이유는 훨씬 더 많은 computation을 하기 때문입니다.

--disc_mode: GAN의 discriminator에서 어떤 모델을 사용할 지 입니다. vanila는 MLP레이어로 구성되어있는 심플한 모델이고 selfattention은 self-attention mechanism을 도입한 모델입니다. vanila와 selfattention중에 선택하셔야 합니다. selfattention은 vanila에 비해 트레이닝 속도가 아주 느립니다. 그 이유는 훨씬 더 많은 computation을 하기 때문입니다.

--log_path: 트레이닝 후 결과에 대한 log가 저장되는 path를 설정합니다. 예를들어 path/to/directory/ganmodel_1v 이라는 path를 값으로 준다면 ganmodel_1v 폴더 안에 로그 파일이 만들어 질 것입니다.

--output_path: 트레이닝 후 결과로 나오는 모델파일 등이 output_path에 저장됩니다. 예를들어 path/to/directory/ganmodel_1v/output 이라는 path를 값으로 준다면 트레이닝 완료 후 output path안에 모델파일이 생성이 됩니다.

--device: GPU를 활용할지 CPU를 활용할지 선택하는 부분입니다. cpu를 선택하면 GPU를, cuda를 선택하면 GPU활용하여 모델 트레이닝을 합니다. GPU가 있다면 GPU로 트레이닝을 하는 것을 권장합니다 (훨씬 빠른 트레이닝 시간)

--learning_rate_gen: GAN모델의 generator의 optimizer의 learning rate를 설정하는 부분입니다. learning rate가 너무 크면 트레이닝 시 fine granularity 수준의 minima를 찾기 힘들 수 있고 learning rate가 너무 작으면 트레이닝이 너무 오래걸릴 수 있습니다. 0.0002를 디폴트 값으로 지정하였습니다.

--learning_rate_disc: GAN모델의 discriminator의 optimizer의 learning rate를 설정하는 부분입니다. learning rate가 너무 크면 트레이닝 시 fine granularity 수준의 minima를 찾기 힘들 수 있고 learning rate가 너무 작으면 트레이닝이 너무 오래걸릴 수 있습니다. 0.0002를 디폴트 값으로 지정하였습니다.

--zdim: Generator에서 noise를 생성하여 실제 이미지와 비슷한 것을 생성하려고 할 때 설정하는 noise의 dimension 크기입니다. 100을 디폴트 값으로 설정하였습니다.

--height: 이미지 데이터의 높이 사이즈입니다. 현재 사용중인 현대 이미지 데이터의 높이는 10입니다. 10을 디폴트 값으로 설정하였습니다.

--width: 이미지 데이터의 폭 사이즈입니다. 현재 사용중인 현대 이미지 데이터의 폭는 30입니다. 그러나 --datashape를 dropempty로 설정할 경우 폭이 14가 됩니다. 그러므로 이 경우 무조건 width를 14로 설정해야 합니다. --datashape를 original로 설정할 경우 폭이 30이 됩니다. 그러므로 이 경우 무조건 width를 14로 설정해야 합니다. 디폴트 값은 14입니다.

--input_dim: 이미지 데이터를 flattening 하였을 때 dimension 사이즈입니다. --datashape가 original일때는 3 * 10  * 30이기 때문에 900, --datashape가 dropempty일때는 3 * 10 * 14이기 때문에 420으로 설정해줘야 합니다. 디폴트 값은 420입니다.

--hidden_dim: 모델 트레닝시 hidden layer들의 dimension입니다. 디폴트는 256입니다. 64, 128, 256, 512 등 2의 제곱의 형태로 dimension 사이즈를 결정하는 것이 일반적입니다. 숫자가 크면 클수록 컴퓨팅시 더 시간이 오래걸리고 메모리도 많이 가져갑니다.

--output_dim: discriminator의 output dimension입니다. discriminator는 어떤 image가 들어왔을 때 real image와 fake image 2개의 클래스 중 하나를 예측하는 것이기 때문에 dimension이 1이면 됩니다. 디폴트는 1입니다.

--embed_dim: gen_mode와 disc_mode가 selfattention일 때 각 self-attention sequence들의 embedding dimension을 정해는 부분입니다. 디폴트는 32로 해 놓았습니다. 16, 32, 64, 128, 256, 512 형식으로 선택하는 것을 추천드립니다.

--num_heads: gen_mode와 disc_mode가 selfattention일 때 각 self-attention sequence들의 embedding dimension을 여러개로 (multi-head) 나누어 트레이닝하는 부분입니다. 예를 들어, embed_dim이 32이고 num_heads가 4이면 32/4 4개의 multi-heads가 생기고 각 head는 8의 dimension 사이즈를 가지게 됩니다. 디폴트는 2입니다.

--optimizer: generator와 discriminator의 optimizer로 어떤 것을 사용할 지 정하는 부분입니다. adam, adamw 그리고 rmsprop중에서 결정할 수 있습니다. 디폴트는 adamw입니다.

--weight_clip: loss_mode가 wgan일 때 gradient weight를 일정 값 이상, 이하가 되지 못하게 clipping 하는 부분입니다. 디폴트 값은 0.01입니다.

--batch_size: 트레이닝시 mini batch의 사이즈입니다. 디폴트는 512입니다.

--epochs: 트레이닝 시 몇번의 트레이닝 loop를 돌 것인지 결정하는 부분입니다. 디폴트는 100입니다.

--constraint_lambda: loss term에서 fixed pixel을 구현하기 위해 그 픽셀 부분에 대하여 어느정도 가중치를 줄 것인지 결정하는 부분입니다. 디폴트는 10입니다

--lambda_gp: --loss_mode가 wgangp일 경우 WGAN-GP에서 패널티를 얼마나 강하지 주는지에 대한 가중치를 결정하는 부분입니다. 디폴트는 10입니다.

--critic_iter: --loss_mode가 wgan이거나 wgangp이면 discriminator를 generator보다 더 많이 업데이트를 합니다. critic_iter를 5로 설정한다면 generator의 트레이닝 스텝 한번을 취할 때 discriminator는 5번의 스텝을 취합니다. 디폴트는 5입니다.

--opt_beta1: torch의 adam이나 adamw optimizer는 betas라는 인자가 있는데 여기에 들어갈 첫번째 beta 값을 설정하는 부분입니다. 디폴트는 0.0입니다.

--opt_beta2:torch의 adam이나 adamw optimizer는 betas라는 인자가 있는데 여기에 들어갈 두 번째 beta 값을 설정하는 부분입니다. 디폴트는 0.9입니다.


트레이닝 스크립트 실행 예시:
1.Info-GAN에 wgan loss를 실행하는 예시 wgan loss를 사용한다면 wgan의 weight_clip을 설정해 주어야합니다.
만약 --datashape를 dropempty로 설정한다면 무조건 --width를 14, input_dim을 420으로 설정해 주어야 합니다. 반대로 --datashape를 original로 설정한다면
무조건 --width를 30, input_dim을 900으로 설정해 주어야 합니다.
vanila GAN에는 wgan_loss를 vanila로 설정할 수 없고 무조건 wgangp나 wgan으로 설정해 주어야 합니다.

python main.py --data_path path/to/imagedata --datashape dropempty --gan_model infogan --loss_mode wgan 
--gen_mode vanila --disc_mode vanila --log_path path/to/logdir --output_path path/to/outputpath
--device cuda --learning_rate_gen 0.0002 --learning_rate_disc 0.0002 --zdim 100 --height 10 --width 14
--input_dim 420 --hidden_dim 256 --embed_dim 32 --output_dim 1 --num_heads 2 --optimizer adamw
--weight_clip 0.01 --batch_size 512 --epochs 400 --constraint_lambda 10 --critic_iter 5 --opt_beta1 0.0 --opt_beta2 0.9




2.Vanila-GAN에 WGAN-GP loss를 실행하고 self-attention이 아닌, vanila generator와 discriminator를 사용하는 모델을 사용하는 예시입니다.

python main.py --data_path path/to/imagedata --datashape original --gan_model vanila --loss_mode wgangp 
--gen_mode vanila --disc_mode vanila --log_path path/to/logdir --output_path path/to/outputpath
--device cuda --learning_rate_gen 0.0002 --learning_rate_disc 0.0002 --zdim 100 --height 10 --width 14
--input_dim 420 --hidden_dim 256 --embed_dim 32 --output_dim 1 --num_heads 2 --optimizer adamw
--lambda_gp 10 --batch_size 512 --epochs 200 --constraint_lambda 10 --critic_iter 5 --opt_beta1 0.0 --opt_beta2 0.9


모델을 트레이닝하면서 여러 인자들과 내용들을 설정하신 log_path안에 log_summary라는 파일로 자동으로 기록을 하도록 하였습니다.

모델을 트레이닝하면서 나온 이미지 결과물을 비교할 수 있도록 트래킹을 하기위해 Tensorboard를 활용하였습니다. 이 tensorboard에 대한 이벤트 정보는 설정하신 log_path안에 tensorboard_records디렉토리 안에 저장하도록 하였습니다. tensorboard_records 디렉토리 안에 real 폴더 안에는 실제 이미지가, fake에는 트레이닝을 통해 real 데이터에 대응하여 생성한 이미지의 이벤트 로그가 생성되도록 하였습니다.

output_path를 설정하였으면 그 안에 자동으로 fake와 real 디렉토리가 생깁니다. 그 안에 트레이닝이 되고 가장 마지막 batch를 트레이닝 한 후 inference를 해서 생성된 이미지를 fake 디렉토리에, 그에 상응되는 실제 이미지 데이터를 real에 저장합니다.


GAN models에 대하여:


GAN 모델은 크게 2가지로 나누어서 만들었습니다. 하나는 Info-GAN 모델을 만들었고 다른 하나는 일반적인 MLP모델을 적용한 GAN입니다 (이하 vanila gan).

1.Info-GAN
2.Vanila-GAN

Info-GAN 모델도 모델 내부 구조적으로 두가지 다른 모델을 구현하였습니다. 하나는 일반적인 MLP로 구현한 모델이고(이하 vanila) 다른 하나는 self-attention mechanism을 적용한 모델입니다(이하 self-attention).
Vanila-GAN의 모델의 경우에도 모델 내부 구조적으로 두가지 다른 모델을 구현하였습니다. 하나는 일반적인 MLP로 구현한 모델이고(이하 vanila) 다른 하나는 self-attention mechanism을 적용한 모델입니다(이하 self-attention). 즉 아래와 같이 볼 수 있습니다.

1.Info-GAN + Vanila MLP
2.Info-GAN + Selfattention 
3.Vanila-GAN + Vanila MLP
4.Vanila-GAN + Selfattention mechanism

이렇게 세부적으로 4가지의 모델이 된 시점에서 loss function도 다르게 트레이닝 할 수 있도록 만들었습니다. loss function(손실함수)는 세가지가 있습니다. 하나는 일반적인 plain binary cross entropy를 사용는 것이고(이하 vanila) 다른 하나는 wasserstein loss를 사용하는 wgan(이하 wgan)이고, 나머지 하나는 wasserstein with gradient penalty를 적용한(이하 wgangp) 손실함수 입니다. 그래서 모델을 사용할 때 아래와 같은 구현을 할 수 있습니다. 

1.Info-GAN + Vanila MLP + vanila loss
2.Info-GAN + Vanila MLP + wgan loss
3.Info-GAN + Vanila MLP + wgangp loss
4.Info-GAN + Selfattention + vanila loss
5.Info-GAN + Selfattention + wgan loss
6.Info-GAN + Selfattention + wgangp loss
7.Vanila-GAN + Vanila MLP + vanila loss
8.Vanila-GAN + Vanila MLP + wgan loss
9.Vanila-GAN + Vanila MLP + wgangp loss
10.Vanila-GAN + Selfattention mechanism + vanila loss
11.Vanila-GAN + Selfattention mechanism + wgan loss
12.Vanila-GAN + Selfattention mechanism + wgangp loss


세부적인 모델 차이에서, Vanila-GAN + Vanila MLP 계열의 모델들만 다른 모델들과의 차이를 가지고 있는 부분이 있는데, 바로 fixed_pixel 데이터를 트레이닝에 포함하고 있다는 것입니다. 다른 모델들에서는 fixed_pixel에 대한 값은 포함되지 않았고 loss function에서만 fixed_pixel 값을 참조할 수 있도록만 하였습니다. 현재까지 테스트를 진행한 결과 Vanila-GAN + Vanila MLP처럼 모델안에 직접 fixed_pixel 값도 주고 loss function에도 포함시키는 것이 더 원하는 값을 출력한다는 잠정적인 판단을 하게 되었습니다. 이후 다른 모델에서도 fixed_pixel 데이터를 직접 같이 주어서 테스트를 진행해 보려고 합니다.

self-attention을 사용하는 모델들은 vanila MLP를 사용하는 모델보다 트레이닝 속도가 아주 많이 느립니다. 그래서 아직 제대로 트레이닝을 끝마친 모델이 없는데 시간을 들여서 이후에도 테스트를 해보려고 합니다.



========================================================================================================================================================================================================================================================================================================================================================================================

gan_inference.py

--model_path: 사용할 모델의 path를 말합니다. 예를들어 모델이 path/to/model이라는 디렉토리에 model.pt 이라는 파일이라면 path/to/model/modelname.pt로 model_path를 설정하면 됩니다.

--gen_img_path: 생성된 이미지를 저장할 경로를 말합니다. path/to/save/gen_image로 경로를 설정한다면 path/to/save/gen_image 안에 생성되는 이미지들이 저장될 것입니다.

--num_gen_image: 한번의 값을 주고 몇개의 이미지를 생성해 볼 것인지 결정하는 요소입니다. 예를들어 15를 선택하면 하나의 값을 주어 15개의 이미지를 생성해 냅니다.

--durability:  {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9} 중에서 하나의 값을 선택하여 입력합니다. 

--mass: {2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 2.2, 2.21, 2.22, 2.230, 2.24} 중에서 하나의 값을 선택하여 입력합니다.

--strut1: 첫번째 fixed pixel의 값을 입력합니다.

--strut2: 두번째 fixed pixel의 값을 입력합니다.

--device: 이미지 생성 추론을 할 때 GPU를 사용할지 CPU를 사용할지 결정하는 요소입니다. GPU를 사용하려면 cuda, CPU를 사용하려면 cpu라고 쓰면 됩니다.


gan_inference.py 실행 예시

python gan_inference.py --model_path C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\CGAN\logs\vanilagan_wganloss_vanila_dropempty_150\vanila_model.pt --gen_img_path C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\CGAN\logs\vanilagan_wganloss_vanila_dropempty_150\inference_output --num_gen_image 50 --durability 1.4 --mass 2.14 --strut1 0 0 0 --strut2 0 0 0 --device cuda


