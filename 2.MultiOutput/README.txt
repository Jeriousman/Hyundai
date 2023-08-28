##머신러닝 예시 (Sciki-learn, XGBoost, and lightgbm) 2023-08-14
python main_1.10.py --img_data_path C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data --label_data_path C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\results_send --mode ml --datashape dropempty --train_size 0.8 --validation_size 0.1 --log_path C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\logs\lgb --x_scale one --y_scale minmax --model lgb

##딥러닝 예시(Pytorch Deep Learning Models) 2023-08-14
python main_1.10.py --img_data_path C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data --label_data_path C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\results_send --mode dl --datashape dropempty --train_size 0.8 --validation_size 0.1 --log_path C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\logs\mor_test --x_scale one --y_scale minmax --model mor --batch_size 512 --device cuda --dl_lr 0.001 --epochs 2 --image_dim 420 --label_dim 189 --seq_len 140 --embed_dim 64 --optimizer adam


====================================================================================================================================================================================================================================================================================

MultiOutput Regression Task

이 과제는 주로 Scikit-learn 패키지를 사용하여 진행하였습니다. Scikit-learn에는 다양한 머신러닝 기법이 사용되고 있는 머신러닝 패키지 중에 가장 유명한 패키지 중에 하나입니다. Scikit-learn을 사용하는 가장 큰 이유는 Multioutput warpper를 제공하여 Multi-ouput 태스크 지원을 하기 때문입니다. 몇몇 모델들은 자연적으로 멀티아웃풋을 지원하지만 그렇지 않은 경우에는 multi-output wrapper를 사용하여 태스크를 진행하였습니다. 이 라이브러리에서 multi-output을 위해 두가지 방법을 사용합니다. multi-output wrapper와 chaining wrapper입니다. 이 두가지 논리를 차용하여 파이토치를 사용하여 딥러닝의 방법으로도 트레이닝을 시도하였습니다.
즉, scikit-learn을 통해 일반적인 머신러닝 방법으로 모델 트레이닝을 하거나 파이토치 모델을 통해 딥러닝 방법으로 트레이닝 할 수 있습니다.


모델을 트레이닝하기 위해서는 몇가지 필수적인 인자(argument)들을 설정해 주어야 합니다. 이 필수 인자들은 scikit-learn을 사용하든 pytorch를 사용하든 무조건 입력해야하는 인자입니다. 각 인자들에 대한 설명은 아래와 같습니다. 


--img_data_path: 모델에서 사용할 인풋데이터를 가지고 있는 디렉토리의 경로입니다. 현재 현대에서는 이미화한 데이터가 존재하는데 그 이미지 데이터를 의미합니다. 예를들어, hyundai/image_dir 라는 경로의 디렉토리 안에 존재하는 모든 데이터들이 현대 이미지 데이터면 --img_data_path hyundai/image_dir 로 설정하면 됩니다.  

--label_data_path: --img_data_path와 같이 사용될 정답 데이터의 디렉토리 경로입니다. 현재 현대에서 각 이미지 데이터에 1:1로 대응하는 190개로 이루어진 벡터 데이터를 가지고 있습니다. 그 데이터들을 가지고 있는 디렉토리 경로를 적어주면 됩니다. 예를들어, hyundai/label_dir 라는 경로의 디렉토리 안에 존재하는 모든 데이터들이 현대 이미지 데이터면 --label_data_path hyundai/label_dir 로 설정하면 됩니다.  

--mode: 모델 트레이닝을 위해 scikit-learn 클래시컬 머신러닝, pytorch 딥러닝 중 어떤 것을 쓸지 결정하는 인자입니다. pytorch면 dl, scikit-learn이면 ml을 선택해주세요. 디폴트값은 string값 ml 입니다.

--datashape: original과 dropempty 중에 선택해야합니다. original은 들어온 이미지 높이 10 폭 30 RGB 3을 그대로 데이터로 가져온다는 뜻으로 보시면 되고, dropempty를 선택하시면 검은픽셀(정보가 없는)만 존재하는 컬럼들을 모두 빼고 트레이닝을 할것을 명명하는 것으로 이미지 높이 10 폭 14 RGB 3을 데이터로 가져온다는 뜻입니다.모델 중 self-attention을 사용한 모델로 트레이닝 할 때는 dropempty를 강력하게 추천드립니다 (메모리 이슈 관련). 디폴트값은 dropempty입니다.

 --train_size: 머신러닝 모델은 일반적으로 모델 학습을 위해 데이터를 training/validation/test sets으로 분할합니다. 총 데이터에서 트레이닝 데이터로 몇%나 쓸지 정하는 것입니다. 0.8이면 총 데이터의 80%를 트레이닝 데이터로 쓰겠다는 것입니다. 디폴트 값은  0.8입니다.

--validation_size: 위와같은 방법으로 validation set의 데이터로 총 데이터에서 몇%를 사용할지 정하는 것입니다. 디폴트 값은 float값 0.1입니다.

--log_path: 트레이닝 후 결과에 대한 log가 저장되는 path를 설정합니다. 예를들어 path/to/directory/ganmodel_1v 이라는 path를 값으로 준다면 ganmodel_1v 폴더 안에 로그 파일이 만들어 질 것입니다.

--x_scale: 이미지 데이터를 정규화 전처리하는 방식을 정하는 인자입니다. one, minmax, standard 3가지 중 하나를 선택하거나 그 외의 경우에는 정규화 없이 raw data로 트레이닝 태스크가 진행됩니다.
	one의 경우 이미지를 -1과 1 사이의 값으로 정규화 합니다. minmax의 경우 모든 데이터를 0부터 1 사이의 값으로 정규화합니다. standard의 경우 평균값 0과 표준편차 1의 값을 통해 데이터를 변환합니다.
	one으로 정규화를 하였을 경우에는 scaler가 저장되지 않습니다. 디폴트 값은 one입니다.

--y_scale: 190개의 값으로 이루어진 정답 label 벡터를 정규화 합니다. 일반적으로 label값은 정규화를 하지 않지만 이번에 진행하는 현대의 190개의 값들은 모두 다른 단위(unit)을 가지고 있기 때문에 정규화를 하는 것이 좋을 것이라고 판단되었습니다. minmax, standard 중에 선택할 수 있습니다. minmax나 standard가 아닌 경우 정규화가 없이 raw data로 트레이닝이 진행됩니다. 디폴트는 string값 minmax입니다.

--model: 어떤 모델을 사용하여 multi-output regression 태스크를 진행할 것인지 결정합니다.

	머신러닝 모델 중에서는 	[linear, KNeiReg, Ridge, RCV, Lasso, ElasticNet, RanForestReg, DecisionTree, RANSACReg, HuberReg, GBReg, ABReg, SVR, SGDReg, PAReg, TSReg, HuberRegChained, GBRegChained, 	ABRegChained, SVRChained, SGDRegChained, PARegChained, TSRegChained, lgb, xgb] 중에서 모델을 선택할 수 있습니다. lgb와 xgb가 가장 뛰어난 성능을 보이며 그 중에서도 lgb가 더욱 빠른 것으로 테스트 되었습니다.



	각 모델의 설명은 아래 링크에서 참고 할 수 있습니다. 
	linear: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
	KNeiReg: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
	Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
	RCV: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV
	Lasso: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
	ElasticNet: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
	RanForestReg: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
	DecisionTree: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
	RANSACReg: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
	HuberReg: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
	GBReg: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
	ABReg: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
	SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
	SGDReg: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
	PAReg: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html
	TSReg: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html
	lgb: https://lightgbm.readthedocs.io/en/stable/
	xgb: https://xgboost.readthedocs.io/en/stable/

	Multi-output wrapper 참고: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
	Regressor chaining wrapper 참고: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html


	딥러닝 모델 중에서는  [mor, morsa, mocr, mocrsa] 중에서 모델을 선택할 수 있습니다.

	mor는 파이토치를 사용해 MLP를 구현한 모델입니다.
	
	morsa는 파이토치를 사용해 MLP모델에 self-attention layer를 추가해 더욱 expressive함을 얻기위한 모델입니다. 더욱 많은 computation이 필요합니다.

	mocr는 chaining 방법을 사용한 MLP 모델입니다.

	mocrsa는 chaining 방법을 사용한 MLP 모델에 self-attention layer를 추가해 더욱 expressive함을 얻기 위한 모델입니다. 더욱 많은 computation이 필요합니다.



	
====================================================================================================================================================================================================================================================================================

Pytorch 딥러닝을 위한 필수 인자

--batch_size: 딥러닝을 각 데이터 미니배치 안에 몇개의 데이터를 넣을 것인지 지정합니다. 디폴트 값은 integer로 512입니다.

--device: Pytorch model을 사용한다면, 모델 트레이닝을 위해 GPU를 활성화 할지 지정하는 인자입니다. cpu와 cuda중에 선택합니다. 디폴트 값은 string값으로 cpu입니다. GPU를 사용하고 싶으면 꼭 gpu를 적어주어야 합니다.

--dl_lr: 딥러닝을 위한 옵티마이저의 learning rate를 지정하는 인자입니다. 디폴트는 float값으로 0.001입니다.

--epochs: 딥러닝 트레이닝을 위해 몇번의 트레이닝 repitition을 진행할지 지정하는 인자입니다. 디폴트는 integer값으로 50입니다.

--image_dim: 이미지 데이터를 flattening 하였을 때 dimension 사이즈입니다. --datashape가 original일때는 3 * 10  * 30이기 때문에 900, --datashape가 dropempty일때는 3 * 10 * 14이기 때문에 420으로 설정해줘야 합니다. 디폴트 값은 420입니다.

--label_dim: 이미지 데이터에 대응되는 정답 레이블 데이터의 총 크기를 지정합니다. 190개 값의 벡터이기 때문에 디폴트는 integer값으로 189 (190-1). 만약 실제로 200개의 값이라면 199 (200-1)로 설정해 줍니다. 정답 데이터의 크기가 바뀌지 않는이상 바꿔줄 이유가 없습니다.

--seq_len: --datashape가 original이면 10*30 = 300 즉 300이고 dropempty면 10*14 즉 140입니다. 디폴트는 140입니다.

--embed_dim: deep learning model이 selfattention일 때 각 self-attention sequence들의 embedding dimension을 정해는 부분입니다. 디폴트는 64로 해 놓았습니다. 16, 32, 64, 128, 256, 512 형식으로 선택하는 것을 추천드립니다.

--optimizer: 모델의 optimizer로 어떤 것을 사용할 지 정하는 부분입니다. adam, adamw 그리고 rmsprop중에서 결정할 수 있습니다. 디폴트는 adam입니다.



============================================================================================================================================================================================

	

	


트레이닝 결과물:
1.scikit-learn 머신러닝 모델의 경우:
	predictions.csv: 모델 트레이닝 후 예측 결과물을 csv로 만들어 낸 것
	groundtruth.csv: 예측 결과물에 대응하는 실제 정답 레이블 값을 csv로 만들어 낸 것 (비교하기 위해)
	mean_by_cols.json: 각각의 컬럼 값들을 (0~189) 데이터 포인트를 축으로 평균을 계산한 값들. 즉 각 컬럼마다 평균을 계산하여 총 190개의 값이나오게 됨.
	max_by_cols.json: 각 컬럼들에서 최대값을 뽑아 놓은 값.
	min_by_cols.json: 각 컬럼들에서 최소값을 뽑아 놓은 값.
	stdev_by_cols.json:  각각의 컬럼 값들을 (0~189) 데이터 포인트를 축으로 표준편차를 계산한 값들. 즉 각 컬럼마다 평균을 계산하여 총 190개의 값이나오게 됨.
	rmse_by_cols.json: 각각의 컬럼 값들을 (0~189) 데이터 포인트를 축으로 RMSE를 계산한 값들. 즉 각 컬럼마다 평균을 계산하여 총 190개의 값이나오게 됨.
	error_percentage_by_cols.json: 190개의 벡터에서 실제 정답값과 예측 값의 차이를 각각의 값 0부터 189개에 대해 모든 데이터포인트 값의 차이를 평균내서 평균 오차율을 나타낸 값을 json파일로 저장해 놓은 값
	{filename}.joblib: scikit-learn 모델 트레이닝 후 저장된 트레이닝 된 모델. filename이 model_0이라면 model_0.joblib으로 저장이 됨.
	{filename}: 아무 확장자 없이 {filename}으로 저장된 파일은 이번 트레이닝을 하면서 남긴 로그들이 저장되어 있는 파일입니다. 어떤식으로 트레이닝을 했는지 파악할 수 있는 중요한 정보입니다.
	x_scaler.pkl: 인자중에 x_scale을 minmax나 standard로 지정했을 경우 x_scaler.pkl가 저장되어 이후에 다시 사용 할 수 있도록 합니다.
	y_scaler.pkl: 인자중에 y_scale을 minmax나 standard로 지정했을 경우 y_scaler.pkl가 저장되어 이후에 다시 사용 할 수 있도록 합니다.
	
	
	
	
	