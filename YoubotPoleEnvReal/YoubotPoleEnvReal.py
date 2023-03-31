import numpy as np
import gym
from gym.utils import seeding
from gym import spaces, logger
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import ctypes
import pyrealsense2 as rs
import math as m


class YoubotPoleEnvReal(gym.Env):

    def __init__(self):
        super(YoubotPoleEnvReal, self).__init__()
        self.robotposition = {}
        self.push_force = 0
        self.q = [0.0, 0.0]
        self.q_last = [0.0, 0.0]

        self.theta_max = 40 * np.pi / 180
        self.youbot_pos_max = 2.0

        high = np.array(
            [
                self.youbot_pos_max,
                np.finfo(np.float32).max,
                self.theta_max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.counts = 0
        self.steps_beyond_done = None

        # Connect to ROS
        rospy.init_node('rl_twist')
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('odom', Odometry, self.cbOdom)
        self.rate = rospy.Rate(200)
        print('Connected to ROS.')

        # Initialize G-sensor
        # self.lib = ctypes.CDLL('./gsensor.so')
        # self.lib.initialize.restype = ctypes.c_int
        # self.lib.gsensor.restype = ctypes.c_double
        # self.lib.gsensor.argtypes = [ctypes.c_int]
        # self.ret = self.lib.initialize()
        # print('G-sensor initialized.')
        # Use T265 instead of G-sensor
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        self.pipe.start(cfg)
        self.frames = self.pipe.wait_for_frames()

    def cbOdom(self, data):
        self.robotposition = data.pose.pose.position

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        q = [0.0, 0.0]
        # Position of Youbot plate (X-axis)
        q[0] = self.robotposition.x if self.robotposition else 0.
        # print("Position: ", q[0])
        # Angle of the pole
        # q[1] = self.lib.gsensor(self.ret)
        self.frames = self.pipe.wait_for_frames()
        pose = self.frames.get_pose_frame()
        if pose:
            data = pose.get_pose_data()
            w = data.rotation.w
            x = -data.rotation.z
            y = data.rotation.x
            z = -data.rotation.y

            # pitch =  -m.asin(2.0 * (x*z - w*y)) * 180.0 / m.pi
            roll  =  m.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) + (m.pi / 2)
            # yaw   =  m.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) * 180.0 / m.pi
            
            q[1] = roll - 0.028
            # print("R [rad]: Roll: {0:.7f}".format(q[1]))

        # print("Angle: ", q[1])
        self.q_last = self.q
        self.q = q

        # The action is in [-1.0, 1.0], tuning the values should give us optimal response
        self.push_force = action*0.53

        # Set action
        twist = Twist()
        twist.linear.x = self.push_force[0]
        twist.linear.y = 0
        twist.linear.z = 0

        twist.angular.x = 0 
        twist.angular.y = 0
        twist.angular.z = 0
        self.pub.publish(twist)
        
        done = (q[0] <= -self.youbot_pos_max) or (q[0] >= self.youbot_pos_max) \
                or (q[1] < -self.theta_max) or (q[1] > self.theta_max)
        done = bool(done)

        if not done:
            # Normalizing distance and angle values (33/67)
            reward = (1 - (q[0]**2)/12 - (q[1]**2)/0.73108)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = (1 - (q[0]**2)/12 - (q[1]**2)/0.73108)
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        dt = 0.005
        self.v = [(self.q[0] - self.q_last[0])/dt, (self.q[1] - self.q_last[1])/dt]
        self.state = (self.q[0], self.v[0], self.q[1], self.v[1])
        self.counts += 1

        self.rate.sleep()

        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(self):
        self.counts = 0
        self.push_force = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None

        twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        self.pub.publish(twist)

        return np.array(self.state, dtype=np.float32)
    
    def render(self):
        return None

    def close(self):
        twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        self.pub.publish(twist)
        # self.lib.turnoff()
        self.pipe.stop()
        return None

if __name__ == "__main__":
    env = YoubotPoleEnvReal()
    env.reset()
    try:
        for _ in range(1000):
            action = env.action_space.sample() # random action
            env.step(np.array([0]))
            print(env.state)
    finally:
        env.close()
