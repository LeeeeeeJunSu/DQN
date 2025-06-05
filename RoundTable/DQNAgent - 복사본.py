import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

# 협동 시나리오
is_cooperative = True
action_one = 1.0
action_two = -1.0

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(7, 128) # 관찰 공간: 7개의 입력 (ball_x, ball_y, ball_z, ball_speed_x, ball_speed_y, ball_speed_z, agent_z)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2) # 액션 공간: 2개의 출력 (상, 하)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, replay_buffer_size, warmup_count, batch_size, eps_start, eps_end, eps_decay, network_update_freq, gamma):
        self.replay_buffer_size = replay_buffer_size
        self.warmup_count = warmup_count
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.network_update_freq = network_update_freq
        self.gamma = gamma
        
        self.input_queue = []
        self.output_queue = []
        self.stop_flag = False
    
    def start(self):
        self.stop_flag = False
        self.internal_thread = threading.Thread(target=self.InternalThread, daemon=True)
        self.internal_thread.start()

    def stop(self):
        self.stop_flag = True
        if self.internal_thread.is_alive():
            self.internal_thread.join()
    
    def get_action(self, ball_x, ball_y, ball_z, ball_speed_x, ball_speed_y, ball_speed_z, agent_z, agent_distance_to_ball, episode_done):
        self.input_queue.append((ball_x, ball_y, ball_z, ball_speed_x, ball_speed_y, ball_speed_z, agent_z, agent_distance_to_ball, episode_done))
        while len(self.output_queue) < 1:
            time.sleep(0.001)
        return self.output_queue.pop(0) 
    
    def get_state(self):
        while len(self.input_queue) < 1:
            time.sleep(0.001)
        return self.input_queue.pop(0)

    def InternalThread(self):

        # 1. 네트워크 및 최적화 관련 초기화 (로컬 변수)
        q_network = Network()
        target_network = Network()
        target_network.load_state_dict(q_network.state_dict())
        target_network.eval()
        optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3)

        # 2. Replay Buffer 및 정규화 통계 준비
        replay_buffer = deque(maxlen=self.replay_buffer_size)
        warmup_states = []
        state_mean = None
        state_std = None

        # 3. Exploration 파라미터
        epsilon = self.eps_start
        update_count = 0
        learn_count = 0

        # 4. 상태 저장용 변수
        prev_state = None
        prev_action = None

        # 5. 학습 루프
        while not self.stop_flag:
            # Supervisor에서 action 후 결과 도착
            ball_x, ball_y, ball_z, spd_x, spd_y, spd_z, agent_z, distance, done = self.get_state()
            state = np.array([ball_x, ball_y, ball_z, spd_x, spd_y, spd_z, agent_z], dtype=np.float32)

            # 시나리오 1 - 공이 중심에 가까워질수록 보상 증가
            if is_cooperative:
                reward = -np.sqrt(ball_x ** 2 + ball_y ** 2 + ball_z ** 2)
            # 시나리오 2 - 공이 자신과 가까워질수록 보상 증가, 공이 떨어지는 것은 상관 없음
            else:
                reward = -distance

            # Logging reward, epsilon
            print(f"Update count: {update_count}, Reward: {reward:.4f}, Epsilon: {epsilon:.4f}")

            # warm-up 중이면 랜덤 행동 + 상태 저장
            if state_mean is None:
                if len(warmup_states) < self.warmup_count:
                    action = random.choice([0, 1])
                    self.output_queue.append(action_one if action == 0 else action_two)

                    if prev_state is not None:
                        replay_buffer.append((prev_state, prev_action, reward, state, done))
                        warmup_states.append(prev_state)

                    prev_state = state
                    prev_action = action
                    continue

                # warm-up 끝났으면 정규화 통계 계산 (1회만)
                warmup_np = np.array(warmup_states)
                state_mean = warmup_np.mean(axis=0)
                state_std = warmup_np.std(axis=0) + 1e-8
                print(f"Warmup complete.")

            # replay buffer에 transition 저장
            if prev_state is not None and prev_action is not None:
                replay_buffer.append((prev_state, prev_action, reward, state, done))

            # ε-greedy 정책으로 행동 선택
            norm_state = (state - state_mean) / state_std
            state_tensor = torch.tensor([norm_state], dtype=torch.float32)

            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    q_values = q_network(state_tensor)
                    action = int(torch.argmax(q_values).item())
                    
            # 현재 상태를 다음 루프를 위한 prev로 저장
            prev_state = state
            prev_action = action

            self.output_queue.append(action_one if action == 0 else action_two)

            update_count += 1

            # 학습 조건 충족 시 학습 수행
            if update_count % self.batch_size == 0:
                print (f"Update count: {update_count}")
                batch = random.sample(replay_buffer, self.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor([(s - state_mean) / state_std for s in states], dtype=torch.float32)
                next_states = torch.tensor([(s - state_mean) / state_std for s in next_states], dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                q_values = q_network(states).gather(1, actions)
                with torch.no_grad():
                    next_q = target_network(next_states).max(1)[0].unsqueeze(1)
                    targets = rewards + (1 - dones) * self.gamma * next_q

                loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                print(f"Loss: {loss.item():.4f}")
                optimizer.step()
                learn_count += 1

                if learn_count % self.network_update_freq == 0:
                    target_network.load_state_dict(q_network.state_dict())
                    print(f"Target network updated at learn count: {learn_count}")
            
            # ε-greedy 탐색률 업데이트
            epsilon = max(self.eps_end, epsilon * self.eps_decay)