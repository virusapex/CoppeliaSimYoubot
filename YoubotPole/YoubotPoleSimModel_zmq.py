from zmqRemoteApi import RemoteAPIClient


class YoubotPoleSimModel():

    def __init__(self, name='YoubotPole'):
        self.name = name
        self.client_ID = None

        self.block_handle = None
        self.revolute_joint_handle = None
        # front left, rear left, rear right, front right
        self.wheelJoints=[-1,-1,-1,-1]
        self.pole_handle = None

    def initializeSimModel(self, sim):
        self.block_handle = sim.getObject('/youBot/Block')
        if (self.block_handle != -1):
            print('Got the block handle.')

        self.revolute_joint_handle = sim.getObject('/youBot/Revolute_joint')
        if (self.revolute_joint_handle != -1):
            print('Got the revolute joint handle.')

        self.wheelJoints[0] = sim.getObject('./rollingJoint_fl')
        self.wheelJoints[1] = sim.getObject('./rollingJoint_rl')
        self.wheelJoints[2] = sim.getObject('./rollingJoint_rr')
        self.wheelJoints[3] = sim.getObject('./rollingJoint_fr')
        if (self.wheelJoints[3] != -1):
            print('Got the wheel joint handles.')

        q = sim.getObjectPosition(self.block_handle, -1)
        q = sim.getJointPosition(self.revolute_joint_handle)

        q = sim.getJointPosition(self.wheelJoints[0])
        q = sim.getJointPosition(self.wheelJoints[1])
        q = sim.getJointPosition(self.wheelJoints[2])
        q = sim.getJointPosition(self.wheelJoints[3])

        self.pole_handle = sim.getObject('/youBot/Pole')
        if (self.pole_handle != -1):
            print('Got the pole handle.')
        # Set the initialized position for each joint
        self.setYoubotTorque(sim, 0)
    
    def getJointPosition(self, sim, joint_name):
        q = 0
        if joint_name == 'block':
            q = sim.getObjectPosition(self.block_handle, -1)
        elif joint_name == 'revolute_joint':
            q = sim.getJointPosition(self.revolute_joint_handle)
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return q

    def setYoubotTorque(self, sim, forwBackVel, leftRightVel=0., rotVel=0.):
        sim.setJointTargetForce(self.wheelJoints[0], -forwBackVel-leftRightVel-rotVel)
        sim.setJointTargetForce(self.wheelJoints[1], -forwBackVel+leftRightVel-rotVel)
        sim.setJointTargetForce(self.wheelJoints[2], -forwBackVel-leftRightVel+rotVel)
        sim.setJointTargetForce(self.wheelJoints[3], -forwBackVel+leftRightVel+rotVel)
    
    def addRandomForce(self, sim, force=0.):
        sim.addForce(self.pole_handle, [0,0,0.4], [force,0,0])
