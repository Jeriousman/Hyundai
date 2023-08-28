
##############################################################################################################################################################################################################################################################################################################################
##Scikit-learn, XGB, 그리고 LGB 머신러닝 태스크 하이퍼파라메터 튜닝 예시
python hptuning.py --img_data_path path/to/data --label_data_path path/to/label_data --mode ml --datashape dropempty --train_size 0.8 --validation_size 0.1 --log_path path/to/log --filename lgb_model --model lgb --image_dim 420 --label_dim 189 --x_scale one --y_scale minmax --max_evals 30
##############################################################################################################################################################################################################################################################################################################################

--img_data_path: 모델에서 사용할 인풋데이터를 가지고 있는 디렉토리의 경로입니다. 현재 현대에서는 이미화한 데이터가 존재하는데 그 이미지 데이터를 의미합니다. 예를들어, hyundai/image_dir 라는 경로의 디렉토리 안에 존재하는 모든 데이터들이 현대 이미지 데이터면 --img_data_path hyundai/image_dir 로 설정하면 됩니다.

--label_data_path: --img_data_path와 같이 사용될 정답 데이터의 디렉토리 경로입니다. 현재 현대에서 각 이미지 데이터에 1:1로 대응하는 190개로 이루어진 벡터 데이터를 가지고 있습니다. 그 데이터들을 가지고 있는 디렉토리 경로를 적어주면 됩니다. 예를들어, hyundai/label_dir 라는 경로의 디렉토리 안에 존재하는 모든 데이터들이 현대 이미지 데이터면 --label_data_path hyundai/label_dir 로 설정하면 됩니다. 

--mode: 모델 트레이닝을 위해 scikit-learn 클래시컬 머신러닝, pytorch 딥러닝 중 어떤 것을 쓸지 결정하는 인자입니다. pytorch면 dl, scikit-learn이면 ml을 선택해주세요. 디폴트값은 string값 ml 입니다.

--datashape: original과 dropempty 중에 선택해야합니다. original은 들어온 이미지 높이 10 폭 30 RGB 3을 그대로 데이터로 가져온다는 뜻으로 보시면 되고, dropempty를 선택하시면 검은픽셀(정보가 없는)만 존재하는 컬럼들을 모두 빼고 트레이닝을 할것을 명명하는 것으로 이미지 높이 10 폭 14 RGB 3을 데이터로 가져온다는 뜻입니다.모델 중 self-attention을 사용한 모델로 트레이닝 할 때는 dropempty를 강력하게 추천드립니다 (메모리 이슈 관련). 디폴트값은 dropempty입니다.

--train_size: 머신러닝 모델은 일반적으로 모델 학습을 위해 데이터를 training/validation/test sets으로 분할합니다. 총 데이터에서 트레이닝 데이터로 몇%나 쓸지 정하는 것입니다. 0.8이면 총 데이터의 80%를 트레이닝 데이터로 쓰겠다는 것입니다. 디폴트 값은  0.8입니다.

--validation_size: 위와같은 방법으로 validation set의 데이터로 총 데이터에서 몇%를 사용할지 정하는 것입니다. 디폴트 값은 float값 0.1입니다.

--log_path: 트레이닝 후 결과에 대한 log가 저장되는 path를 설정합니다. 예를들어 path/to/directory/ganmodel_1v 이라는 path를 값으로 준다면 ganmodel_1v 폴더 안에 로그 파일이 만들어 질 것입니다.

--filename: --log_path안에 각 모델마다 따로 결과를 저장해야 하기 때문에 각 모델 결과를 저장할 파일명(디렉토리명)을 지정해줘야합니다. 예를들어 pytorch_model을 --filename으로 지정하고 위의 --log_path를 /hyundai/logs로 지정하였다면 /hyundai/logs/pytorch_model로 이번 트레이닝 결과물들이 저장될 것입니다.

--model: Hyper-parameter tuning을 진행 할 모델명을 입력해야 합니다. 

	머신러닝 모델 중에서는 	[linear, KNeiReg, Ridge, RCV, Lasso, ElasticNet, RanForestReg, DecisionTree, RANSACReg, HuberReg, GBReg, ABReg, SVR, SGDReg, PAReg, TSReg, HuberRegChained, GBRegChained, 	ABRegChained, SVRChained, SGDRegChained, PARegChained, TSRegChained, lgb, xgb] 중에서 모델을 선택할 수 있습니다. lgb와 xgb가 가장 뛰어난 성능을 보이며 그 중에서도 lgb가 더욱 빠른 것으로 테스트 되었습니다.

	딥러닝 모델 중에서는  [mor, morsa, mocr, mocrsa] 중에서 모델을 선택할 수 있습니다.


--image_dim: 이미지 데이터를 flattening 하였을 때 dimension 사이즈입니다. --datashape가 original일때는 3 * 10  * 30이기 때문에 900, --datashape가 dropempty일때는 3 * 10 * 14이기 때문에 420으로 설정해줘야 합니다. 디폴트 값은 420입니다.

--label_dim: 이미지 데이터에 대응되는 정답 레이블 데이터의 총 크기를 지정합니다. 190개 값의 벡터이기 때문에 디폴트는 integer값으로 189 (190-1). 만약 실제로 200개의 값이라면 199 (200-1)로 설정해 줍니다. 정답 데이터의 크기가 바뀌지 않는이상 바꿔줄 이유가 없습니다.

--x_scale: 이미지 데이터를 정규화 전처리하는 방식을 정하는 인자입니다. one, minmax, standard 3가지 중 하나를 선택하거나 그 외의 경우에는 정규화 없이 raw data로 트레이닝 태스크가 진행됩니다.
	one의 경우 이미지를 -1과 1 사이의 값으로 정규화 합니다. minmax의 경우 모든 데이터를 0부터 1 사이의 값으로 정규화합니다. standard의 경우 평균값 0과 표준편차 1의 값을 통해 데이터를 변환합니다.
	one으로 정규화를 하였을 경우에는 scaler가 저장되지 않습니다. 

--y_scale: 190개의 값으로 이루어진 정답 label 벡터를 정규화 합니다. 일반적으로 label값은 정규화를 하지 않지만 이번에 진행하는 현대의 190개의 값들은 모두 다른 단위(unit)을 가지고 있기 때문에 정규화를 하는 것이 좋을 것이라고 판단되었습니다.
	minmax, standard 중에 선택할 수 있습니다. minmax나 standard가 아닌 경우 정규화가 없이 raw data로 트레이닝이 진행됩니다. 디폴트는 string값 minmax입니다.

--max_evals: Pytorch 모델에서 이 인자를 사용할 경우 하이퍼파라메터 튜닝 한번 시도할 떄  최대 몇번까지 진행 할 것인지를 정하는 인자로 쓰입니다. 예를들어 30이라고 지정한다면 최대 30번까지 테스트 후  trial을 멈춥니다.  pytorch 모델이 아닌 모델에서 사용 할 경우 최대 30개의 모델을 시도해 보고 결과를 달라고 하는 것입니다. 즉 30개의 테스트 중 가장 좋은 결과를 내보내게 됩니다. 이 max_evals의 숫자가 크면 클수록 훨씬 많은 컴퓨팅 시간이 소요됩니다. 




##############################################################################################################################################################################################################################################################################################################################
##Pytorch Deep Learning model 태스크 하이퍼파라메터 튜닝 예시
python hptuning.py --img_data_path path/to/data --label_data_path path/to/label_data --mode dl --datashape dropempty --train_size 0.8 --validation_size 0.1 --log_path path/to/log --filename lgb_model --model mor --image_dim 420 --label_dim 189 --x_scale one --y_scale minmax --max_evals 30 --num_samples 1 --gpus_per_trial 1 --cpus_per_trial 16
##############################################################################################################################################################################################################################################################################################################################


###Pytorch Deep Learning 모델을 사용할 때만 아래 arguments를 설정해 주면 됩니다.

--num_sample: N개 복수만큼 trial 갯수를 실행하고 싶으면 N을 설정해 주어야 합니다. 참고: https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html

--gpus_per_trial: Ray에서 pytorch 모델을 하이퍼파라메터 튜닝시 한번의 trial에서 몇개의 gpu를 사용할 것인지 결정합니다. 디폴트 값은 float값으로 0입니다. 만약 0.25로 설정한다면 각 trial 마다 0.25개의 gpu를 사용하겠다는 것 입니다. 이렇게 fractional하게 gpu를 사용하여 concurrent하게 trials를 실행해 볼 수 있습니다. 만약 1개의 gpu만 있고 gpus_per_trial를 0.25로 설정하면 4개의 concurrent한 trial이 진행됩니다.

--cpus_per_trial: Ray에서 pytorch 모델을 하이퍼파라메터 튜닝시 한번의 trial에서 몇개의 cpu를 사용할 것인지 결정합니다. gpu를 사용하지 않는 경우 cpu를 사용하여 진행합니다. 만약 cpus_per_trial를 4로 지정했고 현재 머신에 4개의 cpu가 존재하면 1개의 trial만 실행이 되고 그 하나의 trial에 4개의 cpu가 모두 사용됩니다. 


