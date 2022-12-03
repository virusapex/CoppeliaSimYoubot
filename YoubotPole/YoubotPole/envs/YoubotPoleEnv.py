import numpy as np
import gym
from gym.utils import seeding
from gym import spaces, logger
import time, copy
from zmqRemoteApi import RemoteAPIClient
from YoubotPole.envs.YoubotPoleSimModel import YoubotPoleSimModel


class YoubotPoleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, port):
        super(YoubotPoleEnv, self).__init__()
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
        self.applied_force = 0

        # Connect to CoppeliaSim
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.getObject('sim')
        print('Connected to remote API server.')
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.client.setStepping(True)
        self.sim.startSimulation()

        self.youbot_pole_sim_model = YoubotPoleSimModel()
        self.youbot_pole_sim_model.initializeSimModel(self.sim)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        q = [0.0, 0.0]
        # Position of Youbot plate (Y-axis)
        q[0] = self.youbot_pole_sim_model.getJointPosition(self.sim, 'block')[1]
        # Adding Gaussian noise to simulate sensor readings
        q[0] += np.random.normal(0,0.003)
        # Angle of the pole
        q[1] = self.youbot_pole_sim_model.getJointPosition(self.sim, 'revolute_joint')
        # Adding Gaussian noise to simulate sensor readings
        q[1] += np.random.normal(0,0.005)
        self.q_last = self.q
        self.q = q

        # bruh = np.random.randint(2000,2400)
        # Adding random force to the pole
        if self.counts == 0:
            self.youbot_pole_sim_model.addRandomForce(self.sim, np.random.choice([-10.,10.]))
        # elif self.counts % bruh == 0 and self.counts - self.applied_force > 400:
        #     if q[0] > 0.:
        #         self.youbot_pole_sim_model.addRandomForce(self.sim, -10.)
        #         self.applied_force = copy.deepcopy(self.counts)
        #     else:
        #         self.youbot_pole_sim_model.addRandomForce(self.sim, 10.)
        #         self.applied_force = copy.deepcopy(self.counts)

        # The action is in [-1.0, 1.0], therefore the force is in [-30, 30]
        self.push_force = action*30

        # Set action
        self.youbot_pole_sim_model.setYoubotTorque(self.sim, self.push_force[0])
        
        done = (q[0] <= -self.youbot_pos_max) or (q[0] >= self.youbot_pos_max) \
                or (q[1] < -self.theta_max) or (q[1] > self.theta_max)
        done = bool(done)

        if not done:
            # Normalizing distance and angle values to be exact equal factors (50/50)
            reward = (1 - (q[0]**2)/8 - (q[1]**2)/0.97478)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = (1 - (q[0]**2)/8 - (q[1]**2)/0.97478)
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

        self.client.step()

        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(self):
        self.counts = 0
        self.push_force = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1) # ensure the Coppeliasim is stopped
        
        # Allows to turn off visualization
        # vrep_sim.simxSetBoolParam(
        #     self.cart_pole_sim_model.client_ID,
        #     vrep_sim.sim_boolparam_display_enabled,
        #     False,
        #     vrep_sim.simx_opmode_oneshot)

        self.client.setStepping(True)
        self.sim.startSimulation()
        self.youbot_pole_sim_model.setYoubotTorque(self.sim, 0)

        return np.array(self.state, dtype=np.float32)
    
    def render(self):
        return None

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        return None

if __name__ == "__main__":
    env = YoubotPoleEnv(23000)
    env.reset()

    for _ in range(5000):
        action = env.action_space.sample() # random action
        env.step(action)
        print(env.state)

    env.close()
