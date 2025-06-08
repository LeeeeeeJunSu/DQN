import socket
import json
from time import sleep
from DQNAgent import DQNAgent
import threading

if __name__ == '__main__':
    agent_count = 4

    agent_list = []
    for i in range(agent_count):
        agent = DQNAgent(
            replay_buffer_size=10000,
            warmup_count=1000,
            batch_size=64,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=0.999,
            network_update_freq=20,
            gamma=0.99,
            agent_name=f"agent_{i}",
            reward_scale=10.0,
            norm_alpha=0.01,
        )
        agent.start()
        agent_list.append(agent)

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_instance:
            socket_instance.bind(('127.0.0.1', 9000 + agent_count))
            socket_instance.listen()
            socket_instance.settimeout(1.0)
            try:
                conn, addr = socket_instance.accept()
                with conn:
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        # Parse the received data
                        data_dict = json.loads(data.decode())
                        ball_x = data_dict['ball_x']
                        ball_y = data_dict['ball_y']
                        ball_z = data_dict['ball_z']
                        ball_speed_x = data_dict['ball_speed_x']
                        ball_speed_y = data_dict['ball_speed_y']
                        ball_speed_z = data_dict['ball_speed_z']
                        gripper_z_list = [data_dict.get(f'agent_gripper_z_{i}', 0.0) for i in range(agent_count)]
                        done = data_dict['IsDone']

                        # Get actions from each agent
                        action_list = []
                        for i in range(agent_count):
                            action_list.append(agent_list[i].get_action(ball_x, ball_y, ball_z, ball_speed_x, ball_speed_y, ball_speed_z, gripper_z_list, done))

                        # Send Response
                        response = json.dumps(action_list)
                        conn.sendall(response.encode('utf-8'))
            except socket.timeout:
                continue



