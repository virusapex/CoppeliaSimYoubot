import sys
sys.path.append('../VREP_RemoteAPIs')
import sim as vrep_sim


class YoubotPoleSimModel():

    def __init__(self, name='YoubotPole'):
        self.name = name
        self.client_ID = None

        self.prismatic_joint_handle = None
        self.revolute_joint_handle = None
        self.wheelJoints=[-1,-1,-1,-1] # front left, rear left, rear right, front right

    def initializeSimModel(self, client_ID):
        try:
            print ('Connected to remote API server')
            client_ID != -1
        except:
            print ('Failed connecting to remote API server')

        self.client_ID = client_ID

        return_code, self.prismatic_joint_handle = vrep_sim.simxGetObjectHandle(
            client_ID, './ME_Platfo2_sub1', vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('get object prismatic joint ok.')

        return_code, self.revolute_joint_handle = vrep_sim.simxGetObjectHandle(
            client_ID, 'revolute_joint', vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('get object revolute joint ok.')

        return_code, self.wheelJoints[0] = vrep_sim.simxGetObjectHandle(
            client_ID, './rollingJoint_fl', vrep_sim.simx_opmode_blocking)
        return_code, self.wheelJoints[1] = vrep_sim.simxGetObjectHandle(
            client_ID, './rollingJoint_rl', vrep_sim.simx_opmode_blocking)
        return_code, self.wheelJoints[2] = vrep_sim.simxGetObjectHandle(
            client_ID, './rollingJoint_rr', vrep_sim.simx_opmode_blocking)
        return_code, self.wheelJoints[3] = vrep_sim.simxGetObjectHandle(
            client_ID, './rollingJoint_fr', vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('get wheel joints ok.')

        # Get the joint position
        #return_code, q = vrep_sim.simxGetJointPosition(self.client_ID, self.prismatic_joint_handle, vrep_sim.simx_opmode_streaming)
        return_code, q = vrep_sim.simxGetObjectPosition(self.client_ID, self.prismatic_joint_handle, -1, vrep_sim.simx_opmode_streaming)
        return_code, q = vrep_sim.simxGetJointPosition(self.client_ID, self.revolute_joint_handle, vrep_sim.simx_opmode_streaming)

        #return_code, q = vrep_sim.simxGetJointPosition(self.client_ID, self.wheelJoints, vrep_sim.simx_opmode_streaming)
        return_code, q = vrep_sim.simxGetJointPosition(self.client_ID, self.wheelJoints[0], vrep_sim.simx_opmode_streaming)
        return_code, q = vrep_sim.simxGetJointPosition(self.client_ID, self.wheelJoints[1], vrep_sim.simx_opmode_streaming)
        return_code, q = vrep_sim.simxGetJointPosition(self.client_ID, self.wheelJoints[2], vrep_sim.simx_opmode_streaming)
        return_code, q = vrep_sim.simxGetJointPosition(self.client_ID, self.wheelJoints[3], vrep_sim.simx_opmode_streaming)
        # Set the initialized position for each joint
        self.setYoubotTorque(0)
    
    def getJointPosition(self, joint_name):
        q = 0
        if joint_name == 'prismatic_joint':
            return_code, q = vrep_sim.simxGetObjectPosition(self.client_ID, self.prismatic_joint_handle, -1, vrep_sim.simx_opmode_buffer)
        elif joint_name == 'revolute_joint':
            return_code, q = vrep_sim.simxGetJointPosition(self.client_ID, self.revolute_joint_handle, vrep_sim.simx_opmode_buffer)
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return q

    def setYoubotTorque(self, forwBackVel, leftRightVel=0., rotVel=0.):
        vrep_sim.simxPauseCommunication(self.client_ID, True)

        vrep_sim.simxSetJointTargetVelocity(
            self.client_ID,
            self.wheelJoints[0],
            -forwBackVel-leftRightVel-rotVel,
            vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(
            self.client_ID,
            self.wheelJoints[1],
            -forwBackVel+leftRightVel-rotVel,
            vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(
            self.client_ID,
            self.wheelJoints[2],
            -forwBackVel-leftRightVel+rotVel,
            vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(
            self.client_ID,
            self.wheelJoints[3],
            -forwBackVel+leftRightVel+rotVel,
            vrep_sim.simx_opmode_oneshot)
        #vrep_sim.simxSetJointMaxForce(self.client_ID, self.wheelJoints[0], 50, vrep_sim.simx_opmode_oneshot)
        #vrep_sim.simxSetJointMaxForce(self.client_ID, self.wheelJoints[1], 50, vrep_sim.simx_opmode_oneshot)
        #vrep_sim.simxSetJointMaxForce(self.client_ID, self.wheelJoints[2], 50, vrep_sim.simx_opmode_oneshot)
        #vrep_sim.simxSetJointMaxForce(self.client_ID, self.wheelJoints[3], 50, vrep_sim.simx_opmode_oneshot)

        vrep_sim.simxPauseCommunication(self.client_ID, False)
