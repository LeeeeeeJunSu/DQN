from controller import Robot
import socket

# 기본 정의
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# 사전정의
min_position = 0.20
max_position = 0.35

# Supervisor <-> Robot 소켓 통신 열기 
supervisor_communicator = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
supervisor_communicator.bind(('127.0.0.1', int(robot.getName()) + 9000))
supervisor_communicator.listen()
supervisor_communicator_conn, addr = supervisor_communicator.accept()

# 디바이스 가져오기
linear_device = robot.getDevice('linear motor')
sensor_device = robot.getDevice('position sensor')
gripper_device = robot.getDevice('motor::left finger')

# 디바이스 초기 설정
linear_device.setPosition(float('inf'))
linear_device.setVelocity(0.0)
sensor_device.enable(timestep)
gripper_device.setPosition(1.0)

# 로봇 제어
while robot.step(timestep) != -1:
    action = float(supervisor_communicator_conn.recv(1024).decode())
    cur_pos = sensor_device.getValue()
    
    if (cur_pos < min_position) and (action < 0.0) or ((cur_pos > max_position) and (action > 0.0)):
        linear_device.setVelocity(0.0)
    else:
        linear_device.setVelocity(action)