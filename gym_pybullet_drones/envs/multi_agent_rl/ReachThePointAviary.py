import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

# SPHERE_POS = [60, 10, 10]
SPHERE_POS = [100, 10, 10]
cached_spheres = []
id = 0



class ReachThePointAviary(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.last_drones_dist = [-1000 for _ in range(self.NUM_DRONES)]

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.
        This method is called once per reset, the environment is recreated each time, maybe caching sphere is a good idea(Gyordan)
        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        import pybullet as p
        import csv
        import os
        import experiments.SVS_Code as module_path
        from random import randrange
        env_number = "1" #str(randrange(10))
        csv_file_path = os.path.dirname(
            module_path.__file__) + "/environment_generator/generated_envs/{0}/static_obstacles.csv".format(
            "environment_" + env_number)

        global cached_spheres
        global id

        print("Creating spheres")
        if len(cached_spheres) == 0:
            print("Reading spheres first time")
            with open(csv_file_path, mode='r') as infile:
                reader = csv.reader(infile)
                # prefab_name,pos_x,pos_y,pos_z,radius
                cached_spheres = [[str(rows[0]), float(rows[1]), float(rows[2]), float(rows[3]), float(rows[4])] for rows in
                           reader]

        """
        if id != 0:
            print("Loading from state")
            p.restoreState(id)
        else:
            for sphere in cached_spheres:
                temp = p.loadURDF(sphere[0],
                                  sphere[1:4:],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  physicsClientId=self.CLIENT,
                                  useFixedBase=True,
                                  globalScaling=10 * sphere[4],
                                  flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
                                  )
                p.changeVisualShape(temp, -1, rgbaColor=[0, 0, 1, 1])
            id = p.saveState()
        """
        """
        import pybullet as p
        sphere = p.loadURDF(
            "/home/cam/Desktop/Tutor/SVS/gym-pybullet-drones/experiments/SVS_Code/3D_Models/Hangar/hangar.urdf",
            [0, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT,
            useFixedBase=True,
            globalScaling=1 * 0.5,
        )
        """

    ################################################################################

    def step(self, action):
        return super().step(action)

    def reset(self):
        x = super().reset()
        # todo addcode
        return x

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {} #[0,0,0,0]
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD

        for i in range(0, 2):
            #xorigin_dist = states[i, 0]
            #if xorigin_dist - self.last_drones_dist[i] < 0.4:
            #    rewards[i] = -5
            #else:
            #    self.last_drones_dist[i] = xorigin_dist
            #    rewards[i] = 1 #-1 / (10*xorigin_dist)

            #if self.last_drones_dist[i] > xorigin_dist and self.last_drones_dist[i] - xorigin_dist > 0.2:
            #    self.last_drones_dist[i] = xorigin_dist
            #    rewards[i] = 0.1
            #else:
            #    rewards[i] = -0.02

            # Commenti riguardo il funzionamento del codice
            # states [i, 0] = indica la cordinata x del drone "i" 
            # states [i, 1] = indica la cordinata y del drone "i" 
            # states [i, 2] = indica la cordinata z del drone "i" 

            # Ultimo train lanciato con 500000 martedì 15/11/2022
            # sphere_dist = np.linalg.norm(np.array([states[i, 0], states[i, 1], states[i, 2]]) - SPHERE_POS) ** 2

            # rewards[i] = self.last_drones_dist[i] - sphere_dist
            # self.last_drones_dist[i] = sphere_dist


            # sphere_dist = np.linalg.norm(np.array([states[i, 0], states[i, 1], states[i, 2]]) - SPHERE_POS) ** 2
            # rewards[i] = self.last_drones_dist[i] - sphere_dist
            # self.last_drones_dist[i] = sphere_dist
            
            sphere_dist = np.linalg.norm(np.array([states[i, 0], states[i, 1], states[i, 2]]) - SPHERE_POS) ** 2
            
            
            #if sphere_dist < self.last_drones_dist[i]:
            #    rewards[i] = 20
            #else:
            #    rewards[i] =  - 20

            #self.last_drones_dist[i] = sphere_dist


            # end_dist = 100 - states[i, 0]

            #se last_drone_dist è maggiore di sphere_dist vuol dire che prima era più lontano
            # if self.last_drones_dist[i] > spere_dist:
            #     rewards[i] = self.last_drone_dist[i] - sphere_dist
            # else:

            if self.last_drones_dist[i] < states[i, 0] and states[i, 0] - self.last_drones_dist[i] > 0.1:
                rewards[i] = 1 / (60-states[i, 0])
                self.last_drones_dist[i] = states[i, 0]
                if states[i, 0] > 60:
                    rewards[i] =  rewards[i]*2.5
                elif states[i, 0] > 20:
                    rewards[i] =  rewards[i]*2
                elif states[i, 0] > 10:
                    rewards[i] =  rewards[i]*1.5
            elif self.last_drones_dist[i] > states[i, 0] and self.last_drones_dist[i] - states[i, 0] > 0.1:
                rewards[i] = -0.1
                self.last_drones_dist[i] = states[i, 0]
            
            if states[i, 0] < -10 or  states[i, 1] < -10 or  states[i, 1] > 10 or states[i, 2] > 10 or states[i, 2] < 0.1:
                rewards[i] = -0.1
        
        for i in range(2, self.NUM_DRONES):
            
            if self.last_drones_dist[i] < states[i, 0] and states[i, 0] - self.last_drones_dist[i] > 0.1:
                rewards[i] = 1 / (60-states[i, 0])
                self.last_drones_dist[i] = states[i, 0]
                if states[i, 0] > 60:
                    rewards[i] =  rewards[i]*2.5
                elif states[i, 0] > 20:
                    rewards[i] =  rewards[i]*2
                elif states[i, 0] > 10:
                    rewards[i] =  rewards[i]*1.5
            elif self.last_drones_dist[i] > states[i, 0] and self.last_drones_dist[i] - states[i, 0] > 0.1:
                rewards[i] = -0.1
                self.last_drones_dist[i] = states[i, 0]
            
            if states[i, 0] < -10 or  states[i, 1] < -10 or  states[i, 1] > 10 or states[i, 2] > 10 or states[i, 2] < 0.1:
                rewards[i] = 0
        return rewards

    # def _computeRewardold(self):
    #     """Computes the current reward value(s).

    #     Returns
    #     -------
    #     dict[int, float]
    #         The reward value for each drone.

    #     """

    #     rewards = {}
    #     states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD

    #     for i in range(0, self.NUM_DRONES):
    #         rewards[i] = -1 * np.linalg.norm(
    #             np.array([states[i, 0], states[i, 1], states[i, 2]]) - SPHERE_POS) ** 2
    #     return rewards

    ################################################################################

    """ Group2
    _computeDone viene controllata ad ogni step interno alla singola simulazione
    il controllo attuale su bool_val viene fatto sul tempo totale trascorso, passati n secondi termina lo scenario
    tenere magari per sicurezza un tetto massimo di tempo per evitare run infinite in caso di reward strutturate male
    aggiungere controllo sulle collisioni (sfere e limiti) e in caso di collisione avvenuta segnare su done al relativo indice
    il valore true. contrallare se tutti i valori sono a true allora settare anche __all__ a true per terminare la scena
    """
    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".

        """

        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        #for i in range(0, self.NUM_DRONES):
        #    if done[i] == False and states[i, 0] > -1:
        #        done[i] = False
        #    else:
        #        done[i] = True
        #cont = 0
        #for i in range(0, self.NUM_DRONES):
        #    if done[i] == True:
        #        cont = cont + 1
        #if cont == self.NUM_DRONES:
        #    done["__all__"] = True

        bool_val = True if self.step_counter / self.SIM_FREQ > 60 else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}

        if not True: #not bool_val:
            for i in range(self.NUM_DRONES):
                if states[i,0] < -1:
                    done[i] = True
                else:
                    done[i] = False
            done["__all__"] = all(done.values())
        else:
            done["__all__"] = bool_val  # True if True in done.values() else False

        if done["__all__"]: self.last_drones_dist = [-1000 for _ in range(self.NUM_DRONES)]

        return done

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        #MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        #MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_X = 70
        MAX_Y = 20
        MAX_Z = 15

        MIN_X = -30
        MIN_Y = -20
        MIN_Z = 0

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_x = np.clip(state[0], MIN_X, MAX_X)
        clipped_pos_y = np.clip(state[1], MIN_Y, MAX_Y)
        clipped_pos_z = np.clip(state[2], MIN_Z, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_x,
                                               clipped_pos_y,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_x = clipped_pos_x / MAX_X
        normalized_pos_y = clipped_pos_y / MAX_Y
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_x,
                                      normalized_pos_y,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_x,
                                      clipped_pos_y,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        #if not (clipped_pos_xy == np.array(state[0:2])).all():
        #    print("[WARNING] it", self.step_counter,
        #          "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
        #              state[0], state[1]))
        if not (clipped_pos_x == np.array(state[0])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped x position [{:.2f}]".format(state[0]))
        if not (clipped_pos_y == np.array(state[1])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped y position [{:.2f}]".format(state[1]))

        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                      state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                      state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
