## 폴더 구조
   
Single-Agent/ # 루트 폴더   
	 ├─ result/ # 결과물 폴더   
	 │	├─ DQN_summary.png # DQN 학습 결과 그래프   
	 │	├─ SAC_summary.png # SAC 학습 결과 그래프   
	 │	├─ val_dqn.mp4 # DQN 검증 영상   
	 │	├─ val_sac.mp4 # SAC 검증 영상   
	 │	└─ 설명.txt # 보고서 작성용 설명 파일   
	 │   
	 └─ webots_projects/ # Webots 프로젝트 폴더   
		 ├─ controllers/    
		 │	 └─ roundtable/   
		 │	 	 ├─ best_model_DQN.pth # DQN 최대 성능 모델   
		 │	 	 ├─ best_model_SAC.pth # SAC 최대 성능 모델   
		 │	 	 ├─ DQN_Agent.py # DQN 에이전트 구현   
		 │	 	 ├─ roundtable.py # Webots 컨트롤러 파일   
		 │	 	 ├─ roundtable_DQN.py # DQN 학습 코드   
		 │	 	 ├─ roundtable_SAC.py # SAC 학습 코드   
		 │	 	 ├─ roundtable_VAL.py # 모델 검증 코드   
		 │	 	 └─ SAC_Agent.py # SAC 에이전트 구현   
		 │   
		 └─ worlds/ #    
		 	 ├─ .roundtable.jpg # 썸네일 이미지   
		 	 ├─ .roundtable.wbproj # 프로젝트 설정 파일   
		 	 └─ roundtable.wbt # 시뮬레이션 월드 파일   

----
## 실행 환경
   
IDE: PyCharm 2023 Community Edition   
Python: 3.12   
Webots: R2025a   
라이브러리   
~~~
PyTorch: 2.7.0   
NumPy: 1.26.4   
Matplotlib: 3.8.3   
Pandas: 2.2.2   
~~~
----
## 실행 방법
   
본 프로젝트는 Webots 시뮬레이터와 PyCharm 디버깅 환경에서 실행하였습니다.  
별도의 실행 스크립트는 제공하지 않으며, 아래 절차에 따라 실행합니다.

1. Webots에서 `roundtable.wbt` 월드를 엽니다.
2. 시뮬레이션을 시작하기 전, 컨트롤러 설정이 `roundtable.py`로 되어 있는지 확인합니다. (`extern` 모드 사용)
3. PyCharm에서 `roundtable.py` 파일을 실행합니다.
4. 실행 전, `roundtable_VAL.py` 혹은 학습 코드 내부에 다음과 같이 모드를 설정합니다:
~~~
MODE = "val"       # 또는 "train" (검증 / 학습 모드)
TARGET = "sac"     # 또는 "dqn" (에이전트 종류)
~~~
5. 설정이 완료되면 PyCharm에서 디버깅 모드 또는 실행으로 시뮬레이션을 시작합니다.

----
## 기타
- 학습 결과 및 성능 비교는 `result/` 폴더 내 그래프 및 영상 파일을 참조하십시오.