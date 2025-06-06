from controller import Supervisor
import socket
import time
import json
import random
import math

# 기본 정의
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# 사전정의
agent_count = 8
reset_standard = 0.8

# Supervisor <-> Robot 소켓 통신 열기 
time.sleep(0.2)
robot_communicator_list = []
for i in range(agent_count):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 9000 + i))
    robot_communicator_list.append(sock)

# Supervisor <-> DQNServer 소켓 통신 열기
dqn_server_communicator = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
dqn_server_communicator.connect(('localhost', 9000 + agent_count))

# 관찰할 노드 가져오기
ball = supervisor.getFromDef("BALL")
robot_gripper_list = []
robot_body_list = []
for i in range(agent_count):
    robot_gripper_list.append(supervisor.getFromDef(f"ROBOT_GRIPPER_{i:02}"))
    robot_body_list.append(supervisor.getFromDef(f"ROBOT{i:02}"))

# 공 초기 위치 및 속도 랜덤 설정
radius = random.uniform(0, 0.4)
pos_angle = random.uniform(0, 2 * math.pi)
ball_x = radius * math.cos(pos_angle)
ball_y = radius * math.sin(pos_angle)
ball.getField("translation").setSFVec3f([ball_x, ball_y, ball.getField("translation").getSFVec3f()[2]])

# 공 초기 속도 랜덤 설정
speed = 0.5
angle = random.uniform(0, 2 * math.pi)
vx = speed * math.cos(angle)
vy = speed * math.sin(angle)
ball.setVelocity([vx, vy, 0, 0, 0, 0])

# Robot 제어 및 DQN 학습
while supervisor.step(timestep) != -1:
    # 관찰 데이터 수집
    data_dict = {}
    data_dict['ball_x'] = ball.getField("translation").getSFVec3f()[0]
    data_dict['ball_y'] = ball.getField("translation").getSFVec3f()[1]
    data_dict['ball_z'] = ball.getField("translation").getSFVec3f()[2]
    data_dict['ball_speed_x'] = ball.getVelocity()[0]
    data_dict['ball_speed_y'] = ball.getVelocity()[1]
    data_dict['ball_speed_z'] = ball.getVelocity()[2]
    for i in range(agent_count):
        data_dict[f'agent_z_{i}'] = robot_gripper_list[i].getField("translation").getSFVec3f()[2]
        agent_position_x = robot_body_list[i].getField("translation").getSFVec3f()[0]
        agent_position_y = robot_body_list[i].getField("translation").getSFVec3f()[1]
        distance_to_ball = ((data_dict['ball_x'] - agent_position_x) ** 2 + (data_dict['ball_y'] - agent_position_y) ** 2) ** 0.5
        data_dict[f'agent_distance_to_ball_{i}'] = distance_to_ball
    
    # Ball이 중심으로부터 0.8이상 떨어지면 종료
    data_dict['IsDone'] = data_dict['ball_x'] ** 2 + data_dict['ball_y'] ** 2 > reset_standard

    if(data_dict['IsDone']):
        supervisor.worldReload()
        
    # DQNServer에 현재 상태 전송
    dqn_server_communicator.sendall(json.dumps(data_dict).encode('utf-8'))

    # DQNServer로부터 액션 받기
    action_list = json.loads(dqn_server_communicator.recv(1024).decode('utf-8'))

    # 각 로봇에 액션 전송
    for i in range(agent_count):
        robot_communicator_list[i].sendall(str(action_list[i]).encode('utf-8'))


