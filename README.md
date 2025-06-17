# 원탁 강화학습 시뮬레이션

본 프로젝트는 최우람 작가의 설치 작품 **"원탁"**을 Webots 시뮬레이터에서 강화학습으로 구현하고, 작품의 메시지를 기술적으로 재해석하는 것을 목표로 한다. 싱글 에이전트와 멀티 에이전트 두 가지 시나리오를 구성하여 DQN과 SAC 알고리즘을 적용하였다.

## 폴더 구조
```
RoundTable/           # Root
├─ DQNAgent.py        # 멀티 에이전트용 DQN 에이전트
├─ Server.py          # 멀티 에이전트 학습/추론 서버
├─ controllers/       # Webots 컨트롤러
│   ├─ ROBOT/ROBOT.py          # 각 에이전트 모터 제어
│   └─ SUPERVISOR/SUPERVISOR.py# Server와 통신해 행동 전달
└─ worlds/            # Webots 월드 파일
```

## 실행 환경
- **Python** 3.12
- **Webots** R2025a
- 주요 라이브러리: PyTorch 2.7.1, NumPy 2.3.0

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
