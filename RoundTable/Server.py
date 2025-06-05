import socket
import json
from time import sleep
from DQNAgent import DQNAgent
import threading

if __name__ == '__main__':
    agent_count = 8

    agent_list = []
    for i in range(agent_count):
        agent = DQNAgent(10000, 1000, 50, 1.0, 0.01, 0.999, 5, 0.99, agent_name=f"agent_{i}")
        #agent = DQNAgent(100, 10, 10, 1.0, 0.01, 0.995, 10, 0.99)
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
                        agent_z_list = []
                        agent_distance_to_ball_list = []
                        for i in range(agent_count):
                            agent_z_list.append(data_dict["agent_z_" + str(i)])
                            agent_distance_to_ball_list.append(data_dict["agent_distance_to_ball_" + str(i)])
                        # Get actions from agents
                        action_list = []
                        for i in range(agent_count):
                            action = agent_list[i].get_action(ball_x, ball_y, ball_z, ball_speed_x, ball_speed_y, ball_speed_z, agent_z_list[i], agent_distance_to_ball_list[i], data_dict['IsDone'])
                            action_list.append(action)
                        # Send Response
                        response = json.dumps(action_list)
                        conn.sendall(response.encode('utf-8'))
            except socket.timeout:
                continue



