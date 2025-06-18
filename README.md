# 원탁 강화학습 시뮬레이션
## - 멀티 에이전트 시나리오

## 폴더 구조
```
Root
├─ README.md # 프로그램 및 구현 사항에 대한 설명 기재
├─ requirements.txt # 필요한 라이브러리 목록
└─ RoundTable/
	├─ DQNAgent.py # 멀티 에이전트용 DQN 에이전트
	├─ Server.py # 멀티 에이전트 학습/추론 서버
	├─ controllers/
	│	├─ ROBOT/ROBOT.py # 각 에이전트/모터 제어
	│	└─ SUPERVISOR/SUPERVISOR.py # Server와 통신해 행동 전달
	└─ worlds/ # Webots 월드 파일
```

## 실행 환경
- **Python** 3.13.1
- **Webots** R2025a
- **주요 라이브러리** torch 2.7.1, numpy 2.3.0

## 실행 방법
1. 실행 전 환경 설정
1.1. 필요 라이브러리 설치 전 충돌을 막기 위해 기존 설치 되어 있는 라이브러리를 삭제합니다.
	pip freeze > RemovedPackage.txt
	pip uninstall -y -r RemovedPackage.txt
1.2. 필요 라이브러리를 설치합니다.
	pip install -r requirements.txt
2. 실행
2.1. Server.py 실행
2.2. Webot Simulation 진행
3. 결과 확인
3.1. 스탭 정보: training_logs/agent_%d/Step_Debug.txt
3.1. 학습 정보: training_logs/agent_%d/Learn_Debug.txt
3.1. 에피소드 정보: training_logs/agent_%d/Episode_Debug.txt
3.4. 네트워크 정보: training_logs/agent_%d/network_snapshots
