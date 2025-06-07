
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import os

# 속도 선택을 위한 행동 값 목록
action_values = [1.0, 0.5, 0.0, -0.5, -1.0]

# 모든 에이전트 로그의 기본 디렉터리
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(18, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, replay_buffer_size, warmup_count, batch_size, eps_start,
                 eps_end, eps_decay, network_update_freq, gamma, agent_name="agent",
                 reward_scale=1.0, norm_alpha=0.01):
        self.replay_buffer_size = replay_buffer_size
        self.warmup_count = warmup_count
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.network_update_freq = network_update_freq
        self.gamma = gamma

        # 보상 스케일 조정 및 상태 정규화 업데이트 비율
        self.reward_scale = reward_scale
        self.norm_alpha = norm_alpha

        # 에이전트 이름과 전용 로그 디렉터리 준비
        self.agent_name = agent_name
        self.log_dir = os.path.join(log_dir, agent_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.step_log_path = os.path.join(self.log_dir, "Step_Debug.txt")
        self.learn_log_path = os.path.join(self.log_dir, "Learn_Debug.txt")
        self.episode_log_path = os.path.join(self.log_dir, "Episode_Debug.txt")

        if not os.path.exists(self.step_log_path):
            with open(self.step_log_path, "w") as f:
                f.write("step,episode,reward,epsilon,state_mean,state_std,max_q\n")
        if not os.path.exists(self.learn_log_path):
            with open(self.learn_log_path, "w") as f:
                f.write("learn_step,loss\n")
        if not os.path.exists(self.episode_log_path):
            with open(self.episode_log_path, "w") as f:
                f.write("episode,avg_reward\n")

        self.step_log = open(self.step_log_path, "a")
        self.learn_log = open(self.learn_log_path, "a")
        self.episode_log = open(self.episode_log_path, "a")

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
        self.step_log.close()
        self.learn_log.close()
        self.episode_log.close()

    def get_action(self, ball_x, ball_y, ball_z, ball_speed_x, ball_speed_y, ball_speed_z,
                   agent_positions, gripper_z_list, episode_done):
        """상태 정보를 큐에 전달하여 행동을 요청한다."""
        self.input_queue.append(
            (ball_x, ball_y, ball_z, ball_speed_x, ball_speed_y, ball_speed_z,
             agent_positions, gripper_z_list, episode_done)
        )
        while len(self.output_queue) < 1:
            time.sleep(0.001)
        return self.output_queue.pop(0) 

    def get_state(self):
        while len(self.input_queue) < 1:
            time.sleep(0.001)
        return self.input_queue.pop(0)

    def InternalThread(self):
        q_network = Network()
        target_network = Network()
        target_network.load_state_dict(q_network.state_dict())
        target_network.eval()
        optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3)

        replay_buffer = deque(maxlen=self.replay_buffer_size)
        warmup_states = []
        state_mean = None
        state_var = None
        state_std = None

        epsilon = self.eps_start
        update_count = 0
        learn_count = 0

        prev_state = None
        prev_action = None

        episode_reward = 0
        step_count = 0
        episode_index = 0
        loss_list = []

        while not self.stop_flag:
            ball_x, ball_y, ball_z, spd_x, spd_y, spd_z, agent_positions, gripper_z_list, done = self.get_state()
            state_vals = [ball_x, ball_y, ball_z, spd_x, spd_y, spd_z]
            state_vals.extend(list(agent_positions))
            state_vals.extend(list(gripper_z_list))
            state = np.array(state_vals, dtype=np.float32)
            # nan 값이 있는 경우 Action을 중단 후 계속 진행
            if np.isnan(state).any():
                self.output_queue.append(0.0)
                continue
            
            reward = 0.0
            if prev_state is not None:
                delta_vx = spd_x - prev_state[3]
                delta_vy = spd_y - prev_state[4]
                to_center = np.array([-ball_x, -ball_y], dtype=np.float32)
                norm_center = np.linalg.norm(to_center)
                if norm_center > 0:
                    to_center /= norm_center
                    accel_vec = np.array([delta_vx, delta_vy], dtype=np.float32)
                    reward = float(np.dot(accel_vec, to_center)) * self.reward_scale

            # 스텝별 로그 기록
            mean_str = "" if state_mean is None else ",".join(f"{m:.6f}" for m in state_mean)
            std_str = "" if state_std is None else ",".join(f"{s:.6f}" for s in state_std)

            episode_reward += reward
            step_count += 1

            if state_mean is None:
                if len(warmup_states) < self.warmup_count:
                    action = random.choice(list(range(5)))
                    self.output_queue.append(action_values[action])

                    if prev_state is not None:
                        replay_buffer.append((prev_state, prev_action, reward, state, done))
                        warmup_states.append(prev_state)

                    prev_state = state
                    prev_action = action
                    continue

                warmup_np = np.array(warmup_states)
                state_mean = warmup_np.mean(axis=0)
                state_var = warmup_np.var(axis=0)
                state_std = np.sqrt(state_var) + 1e-8
                # 워밍업 완료 로그

            if prev_state is not None and prev_action is not None:
                replay_buffer.append((prev_state, prev_action, reward, state, done))

            norm_state = (state - state_mean) / state_std
            state_tensor = torch.tensor([norm_state], dtype=torch.float32)
            with torch.no_grad():
                q_network.eval()
                q_values = q_network(state_tensor)
                q_network.train()
                qmax_value = torch.max(q_values).item()

            if random.random() < epsilon:
                action = random.choice(list(range(5)))
            else:
                action = int(torch.argmax(q_values).item())

            prev_state = state
            prev_action = action
            self.output_queue.append(action_values[action])
            update_count += 1

            if state_mean is not None:
                diff = state - state_mean
                state_mean = (1 - self.norm_alpha) * state_mean + self.norm_alpha * state
                state_var = (1 - self.norm_alpha) * state_var + self.norm_alpha * (diff ** 2)
                state_std = np.sqrt(state_var) + 1e-8
                # 정규화 값 갱신

            mean_str = "" if state_mean is None else ",".join(f"{m:.6f}" for m in state_mean)
            std_str = "" if state_std is None else ",".join(f"{s:.6f}" for s in state_std)
            self.step_log.write(
                f"{update_count},{episode_index},{reward:.4f},{epsilon:.4f},{mean_str},{std_str},{qmax_value:.4f}\n"
            )
            self.step_log.flush()

            if len(replay_buffer) >= self.batch_size and update_count % self.batch_size == 0:
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
                optimizer.step()
                self.learn_log.write(f"{learn_count},{loss.item():.4f}\n")
                self.learn_log.flush()
                loss_list.append(loss.item())
                learn_count += 1

                if learn_count % self.network_update_freq == 0:
                    target_network.load_state_dict(q_network.state_dict())

            if done:
                epsilon = max(self.eps_end, epsilon * self.eps_decay)
                avg_reward = episode_reward / step_count if step_count else 0.0
                self.episode_log.write(f"{episode_index},{avg_reward:.4f}\n")
                self.episode_log.flush()

                episode_index += 1
                episode_reward = 0
                step_count = 0
                loss_list.clear()
