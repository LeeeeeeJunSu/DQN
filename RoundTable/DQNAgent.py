
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import os
import logging

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
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)


class DQNAgent:
    def __init__(self, replay_buffer_size, warmup_count, batch_size, eps_start,
                 eps_end, eps_decay, network_update_freq, gamma, agent_name="agent"):
        self.replay_buffer_size = replay_buffer_size
        self.warmup_count = warmup_count
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.network_update_freq = network_update_freq
        self.gamma = gamma

        # 에이전트 이름과 전용 로그 디렉터리 준비
        self.agent_name = agent_name
        self.log_dir = os.path.join(log_dir, agent_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.summary_log_path = os.path.join(self.log_dir, "training_summary.txt")

        if not os.path.exists(self.summary_log_path):
            with open(self.summary_log_path, "w") as f:
                f.write("episode,reward,avg_loss,avg_qmax,epsilon\n")

        # 에이전트별 로거 설정
        self.logger = logging.getLogger(agent_name)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.log_dir, "agent_debug.log"))
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        # 동일 이름의 핸들러 중복 추가 방지
        if not self.logger.handlers:
            self.logger.addHandler(handler)

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
        state_std = None

        epsilon = self.eps_start
        update_count = 0
        learn_count = 0

        prev_state = None
        prev_action = None

        episode_reward = 0
        episode_index = 0
        loss_list = []
        qmax_list = []

        while not self.stop_flag:
            ball_x, ball_y, ball_z, spd_x, spd_y, spd_z, agent_positions, gripper_z_list, done = self.get_state()
            state_vals = [ball_x, ball_y, ball_z, spd_x, spd_y, spd_z]
            state_vals.extend(list(agent_positions))
            state_vals.extend(list(gripper_z_list))
            state = np.array(state_vals, dtype=np.float32)

            reward = 0.0
            if prev_state is not None:
                delta_vx = spd_x - prev_state[3]
                delta_vy = spd_y - prev_state[4]
                to_center = np.array([-ball_x, -ball_y], dtype=np.float32)
                norm_center = np.linalg.norm(to_center)
                if norm_center > 0:
                    to_center /= norm_center
                    accel_vec = np.array([delta_vx, delta_vy], dtype=np.float32)
                    reward = float(np.dot(accel_vec, to_center)) * 1000.0
                
            print(f"Update count: {update_count}, Reward: {reward:.4f}, Epsilon: {epsilon:.4f}")
            self.logger.info(
                f"step={update_count}, reward={reward:.4f}, epsilon={epsilon:.4f}")

            episode_reward += reward

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
                state_std = warmup_np.std(axis=0) + 1e-8
                print(f"Warmup complete.")
                self.logger.info("Warmup complete")

            if prev_state is not None and prev_action is not None:
                replay_buffer.append((prev_state, prev_action, reward, state, done))

            norm_state = (state - state_mean) / state_std
            state_tensor = torch.tensor([norm_state], dtype=torch.float32)

            if random.random() < epsilon:
                action = random.choice(list(range(5)))
            else:
                with torch.no_grad():
                    q_network.eval()
                    q_values = q_network(state_tensor)
                    q_network.train()
                    action = int(torch.argmax(q_values).item())
                    qmax_list.append(torch.max(q_values).item())

            prev_state = state
            prev_action = action
            self.output_queue.append(action_values[action])
            update_count += 1

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
                self.logger.info(
                    f"learn_step={learn_count}, loss={loss.item():.4f}")
                loss_list.append(loss.item())
                learn_count += 1

                if learn_count % self.network_update_freq == 0:
                    target_network.load_state_dict(q_network.state_dict())
                    self.logger.info(
                        f"target network updated at learn_step={learn_count}")

            if done:
                epsilon = max(self.eps_end, epsilon * self.eps_decay)
                avg_loss = np.mean(loss_list) if loss_list else 0.0
                avg_qmax = np.mean(qmax_list) if qmax_list else 0.0
                log_path = os.path.join(self.log_dir, f"episode_{episode_index:05d}.txt")
                with open(log_path, "w") as f:
                    f.write(f"Episode: {episode_index}\n")
                    f.write(f"Reward: {episode_reward:.4f}\n")
                    f.write(f"Epsilon: {epsilon:.4f}\n")
                    f.write(f"Learn Count: {learn_count}\n")
                    f.write(f"Update Count: {update_count}\n")
                    f.write(f"Average Loss: {avg_loss:.4f}\n")
                    f.write(f"Average Max Q: {avg_qmax:.4f}\n")

                with open(self.summary_log_path, "a") as f:
                    f.write(f"{episode_index},{episode_reward:.4f},{avg_loss:.4f},{avg_qmax:.4f},{epsilon:.4f}\n")

                self.logger.info(
                    f"episode_end index={episode_index} reward={episode_reward:.4f} epsilon={epsilon:.4f}")

                episode_index += 1
                episode_reward = 0
                loss_list.clear()
                qmax_list.clear()
