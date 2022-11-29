import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

# SPHERE_POS = [60, 10, 10]
SPHERE_POS = [100, 10, 10]
SPHERE_IDS = []
cached_spheres = []
distances = np.array()
id = 0

MAX_DELTA_X = 10
MAX_DELTA_Y = 10
MAX_DELTA_Z = 10
MAX_DIST = np.linalg.norm([MAX_DELTA_X, MAX_DELTA_Y, MAX_DELTA_Z])

class ReachThePointAviarySingle(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240, # Provare con 48
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
                self.SPHERE_ID = temp
                p.changeVisualShape(temp, -1, rgbaColor=[0, 0, 1, 1])
            id = p.saveState()
        """
        global SPHERE_IDS
        SPHERE_IDS = {}

        sphere = cached_spheres[10]
        SPHERE_IDS[0] = p.loadURDF(sphere[0],
                              [1.5, 2, .9],
                              p.getQuaternionFromEuler([0, 0, 0]),
                              physicsClientId=self.CLIENT,
                              useFixedBase=True,
                              globalScaling=20 * sphere[4],
                              flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
                              )
        p.changeVisualShape(SPHERE_IDS[0], -1, rgbaColor=[0, 0, 1, 1])

        #for i in range(len(cached_spheres)):
        #    sphere = cached_spheres[i]
        #    SPHERE_IDS[i] = p.loadURDF(sphere[0],
        #                      sphere[1:4:],
        #                      p.getQuaternionFromEuler([0, 0, 0]),
        #                      physicsClientId=self.CLIENT,
        #                      useFixedBase=True,
        #                      globalScaling=10 * sphere[4],
        #                      flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        #                      )
        #    p.changeVisualShape(SPHERE_IDS[i], -1, rgbaColor=[0, 0, 1, 1])
        
        #for sphere in cached_spheres:
        #    temp = p.loadURDF(sphere[0],
        #                      sphere[1:4:],
        #                      p.getQuaternionFromEuler([0, 0, 0]),
        #                      physicsClientId=self.CLIENT,
        #                      useFixedBase=True,
        #                      globalScaling=10 * sphere[4],
        #                      flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        #                      )
        #    SPHERE_IDS = temp
        #    p.changeVisualShape(temp, -1, rgbaColor=[0, 0, 1, 1])

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
        res = super().step(action)

        return res

    def reset(self):
        x = super().reset()
        # todo addcode for reset env file
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

        for i in range(0, self.NUM_DRONES):
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
        import pybullet as p

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

        bool_val = True if self.step_counter / self.SIM_FREQ > 100 else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}

        if not bool_val:
            for i in range(self.NUM_DRONES):
                if states[i, 0] < -10 or  states[i, 1] < -10 or  states[i, 1] > 10 or states[i, 2] > 10:
                    done[i] = True
                else:
                    done[i] = False
                    for i in range (len(SPHERE_IDS)):
                        x = p.getContactPoints(SPHERE_IDS[i], self.DRONE_IDS[0])
                        if x != ():
                            done[i] = True
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

    #################################################
    # group_2 coustom functions
    #################################################

    '''
    Restituisce un array con shape (30,)
    Fatto così: [∆x1, ∆y1, ∆z1, ∆x2, ∆y2, ∆z2, ... , ∆x10, ∆y10, ∆z10]
    Con dentro le distanze, clippate e normalizzate tra -1 e 1, delle 10 sfere più vicine al drone

    Appunto: Siccome le 10 sfere più vicine raramente saranno molto lontane, per definizione, 
    i valori che questa funzione restituisce non saranno uniformemente distribuiti nell'intervallo -1/1. 
    Saranno concentrati attorno allo 0
    Potrebbe essere un problema in fase di learning
    '''
    def _clipAndNormalizeSphere(self):
        self._processDistances()
        clippedAndNormalizedDistanceComponents = np.array([])
        nearest10distances = self.distances[:10]
        print("Coordinates of the 10 nearest spheres")
        print(nearest10distances)
        # Sorto le 10 sfere in ordine di coordinata x
        nearest10distances = nearest10distances[nearest10distances[:, 0].argsort()]
        print("Sorted by X")
        print(nearest10distances)
        for nearestSphereIndex in range(0, len(nearest10distances)):
            deltaX = self._normalize(np.clip(nearest10distances[nearestSphereIndex][1], -MAX_DELTA_X, MAX_DELTA_X), -MAX_DELTA_X, MAX_DELTA_X)
            deltaY = self._normalize(np.clip(nearest10distances[nearestSphereIndex][2], -MAX_DELTA_Y, MAX_DELTA_Y), -MAX_DELTA_Y, MAX_DELTA_Y)
            deltaZ = self._normalize(np.clip(nearest10distances[nearestSphereIndex][3], -MAX_DELTA_Y, MAX_DELTA_Y), -MAX_DELTA_Z, MAX_DELTA_Z)
            clippedAndNormalizedDistanceComponents = np.append(clippedAndNormalizedDistanceComponents, [deltaX, deltaY, deltaZ])

        return clippedAndNormalizedDistanceComponents

    # Mappa un valore dall'intervallo min-max all'intervallo -1/1
    def _normalize(self, val, min, max):
        return ((val-min)/(max-min))*2-1

    '''
    Mi aspetto che l'array delle posizioni delle sfere abbia shape (numSpheres, 4)
    Cioè sia fatto così [ [Xs1, Ys1, Zs1, Rs1], [Xs2, Ys2, Zs2, Rs2], ...  ] (R è il fattore di scala, il raggio si ottiene moltiplicando per 0.03)
    Restituisco un array con shape (numSpheres, 5)
    Contenente, per ogni sfera: 
        - La sua coordinata X della sfera
        - Il delta X, Y e Z tra sfera e drone in valore assoluto
        - La distanza tra il drone e la superficie della sfera (calcolata correttamente, no cubi)
    Cioè fatto così [ [x1, |∆x1|, |∆y1|, |∆z1|, distanceNorm1], [x2, |∆x2|, |∆y2|, |∆z2|, distanceNorm2], ...]
    Sorta i risultati per distanza dal drone crescente
    '''
    def _processDistances(self):  
        unsortedDistances = np.zeros(shape=(len(self.cached_spheres), 5))
        #for droneIndex in range(0, self.NUM_DRONES):
        droneIndex = 0
        stateVector = self._getDroneStateVector(droneIndex)
        for sphereIndex in range(0, len(self.cached_spheres)): 
            dronePositionVector = np.array(stateVector[0:3], dtype='float64')
            spherePositionVector = np.array(self.cached_spheres[sphereIndex][1:4], dtype='float64')
            sphereRadius = 0.03*np.array(self.cached_spheres[sphereIndex][4], dtype='float64')

            distanceComponents = np.absolute(spherePositionVector - dronePositionVector)
            distanceComponentsCubic = distanceComponents - np.array([sphereRadius, sphereRadius, sphereRadius])
                
            distanceFromCenterNorm = np.linalg.norm(distanceComponents)
            distanceFromSurfaceNorm = distanceFromCenterNorm - sphereRadius
            # X della sfera, componenti della distanza dalla superficie del "cubo" e distanza dalla superficie della sfera
            # le doppie (()) servono !
            singleDistance = np.concatenate(([spherePositionVector[0]], distanceComponentsCubic, [distanceFromSurfaceNorm]))
            unsortedDistances[sphereIndex] = singleDistance
            '''
            print("Sphere position")
            print(spherePositionVector)
            print("Sphere radius")
            print(sphereRadius)
            print("Distance components (from center)")
            print(distanceComponents)
            print("Distance components (from cube faces)")
            print(distanceComponentsCubic)
            print("Distance (from center)")
            print(distanceFromCenterNorm)
            print("Distance (from surface)")
            print(distanceFromSurfaceNorm)
            print()
            '''

        print('Distances processed')
        print(unsortedDistances)
        # Sort by the 4th element of the inner arrays (the norm)
        self.distances = unsortedDistances[unsortedDistances[:, 4].argsort()]
        print('Distances sorted by distance from surface')
        print(self.distances)
