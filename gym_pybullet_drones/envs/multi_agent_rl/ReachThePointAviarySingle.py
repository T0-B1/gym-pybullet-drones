import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

SPHERE_IDS = []
ENV_COUNTER = 20 #VARIABILE OGNI QUANTI STEP VIENE RICREATO IL SET DI SFERE

cached_spheres = []
distances = np.array([])
reload_env_counter = ENV_COUNTER
env_number = 1
passedSphere = 0

evaluation = []
MAX_DELTA_X = 60
MAX_DELTA_Y = 30
MAX_DELTA_Z = 30
MAX_DIST = np.linalg.norm([MAX_DELTA_X, MAX_DELTA_Y, MAX_DELTA_Z])


class ReachThePointAviarySingle(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 exp = None,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int =  240,
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
        global evaluation
        evaluation = exp


    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.
        This method is called once per reset, the environment is recreated each time, maybe caching sphere is a good idea(Gyordan)
        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        import csv
        import os
        import experiments.SVS_Code as module_path
        from random import randrange
        #env_number = "2" #str(randrange(10))
        csv_file_path = os.path.dirname(
            module_path.__file__) + "/environment_generator/generated_envs/{0}/static_obstacles.csv".format(
            "environment_" + str(env_number))

        global cached_spheres
        global SPHERE_IDS
        SPHERE_IDS = {}

        if len(cached_spheres) == 0:
            print("Reading spheres from {0}".format(csv_file_path))
            with open(csv_file_path, mode='r') as infile:
                reader = csv.reader(infile)
                # prefab_name,pos_x,pos_y,pos_z,radius
                cached_spheres = [[str(rows[0]), float(rows[1]), float(rows[2]), float(rows[3]), 10 * float(rows[4])] for rows in
                           reader]
            
        
        if evaluation:
            for i in range(len(cached_spheres)):
                sphere = cached_spheres[i]
                SPHERE_IDS[i] = p.loadURDF(sphere[0],
                                  sphere[1:4],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  physicsClientId=self.CLIENT,
                                  useFixedBase=True,
                                  globalScaling=sphere[4],
                                  )
                p.changeVisualShape(SPHERE_IDS[i], -1, rgbaColor=[0, 0, 1, 1])
        else:
            for i in range(len(cached_spheres)):
                sphere = cached_spheres[i]
                SPHERE_IDS[i] = p.loadURDF(sphere[0],
                                  sphere[1:4],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  physicsClientId=self.CLIENT,
                                  flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
                                  useFixedBase=True,
                                  globalScaling=sphere[4],
                                  )
                p.changeVisualShape(SPHERE_IDS[i], -1, rgbaColor=[0, 0, 1, 1])


    ################################################################################

    def step(self, action):
        res = super().step(action)

        return res

    def reset(self):
        global cached_spheres
        global env_number
        global reload_env_counter
        global passedSphere

        passedSphere = 0

        if reload_env_counter > 0:
            reload_env_counter -= 1
        else:
            reload_env_counter = ENV_COUNTER
            cached_spheres = []
            env_number = (env_number + 1) % 10
            print("CHANGING ENV")
            print(env_number)

        x = super().reset()

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

        for droneIndex in range(0, self.NUM_DRONES):
            
            if self.last_drones_dist[droneIndex] < states[droneIndex, 0] and states[droneIndex, 0] - self.last_drones_dist[droneIndex] > 0.1:
                
                rewards[droneIndex] = 10 / (70-states[droneIndex, 0])
                
                self.last_drones_dist[droneIndex] = states[droneIndex, 0]

                """
                if states[droneIndex, 0] > 60:
                    rewards[droneIndex] =  rewards[droneIndex]*3
                elif states[droneIndex, 0] > 45:
                    rewards[droneIndex] =  rewards[droneIndex]*2.5
                elif states[droneIndex, 0] > 30:
                    rewards[droneIndex] =  rewards[droneIndex]*2
                elif states[droneIndex, 0] > 15:
                    rewards[droneIndex] =  rewards[droneIndex]*1.5
                """

                #if np.abs(states[droneIndex,1]) < 5:
                #    rewards[droneIndex] =  rewards[droneIndex]*1.3
                #if np.abs(states[droneIndex,2]-5) < 2:
                #    rewards[droneIndex] =  rewards[droneIndex]*1.3

                distFromCenter = 1 - (np.linalg.norm(
                    self._normalizeTwo(states[droneIndex,1:3], np.array([-10,0]), np.array([10,10])) -
                    self._normalizeTwo(np.array([0,5]), np.array([-10,0]), np.array([10,10]))   ))

                molt = 1 + (0.3*distFromCenter)

                rewards[droneIndex] =  rewards[droneIndex]*molt

            elif self.last_drones_dist[droneIndex] > states[droneIndex, 0] and self.last_drones_dist[droneIndex] - states[droneIndex, 0] > 0.1:
                rewards[droneIndex] = -0.1
                self.last_drones_dist[droneIndex] = states[droneIndex, 0]
                
            

            #penalità per uscita dai limiti
            
            if (p.getContactPoints(self.PLANE_ID, self.DRONE_IDS[0]) != ()):
                    rewards[droneIndex] = -1

            for l in range (len(SPHERE_IDS)):
                x = p.getContactPoints(SPHERE_IDS[l], self.DRONE_IDS[0])
                if x != ():
                    rewards[droneIndex] = -1
            
            global passedSphere
            global cached_spheres
            cont = 0
            
            for j in range(len(cached_spheres)):
                
                #print(cached_spheres[j])
                if cached_spheres[j][1] < states[droneIndex, 0]:
                    cont += 1

            if cont > passedSphere:
                passedSphere = cont
                rewards[droneIndex] = 0.3

            if states[droneIndex, 0] < -20 or states[droneIndex, 1] < -10 or states[droneIndex, 1] > 10 or states[droneIndex, 2] > 10:
                    rewards[droneIndex] = -1
            #if states[droneIndex, 0] < -19 or states[droneIndex, 1] < -9 or states[droneIndex, 1] > 9 or states[droneIndex, 2] > 9 or states[droneIndex, 2] < 0.5:
            #        rewards[droneIndex] = -0.01

            #if rewards != {}: 
            #    print(rewards)
            

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

        bool_val = True if self.step_counter / self.SIM_FREQ > 120 else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}

        if not bool_val:
            for i in range(self.NUM_DRONES):
                if states[i, 0] < -20 or states[i, 0] > 60 or  states[i, 1] < -10 or  states[i, 1] > 10 or states[i, 2] > 10:
                    done[i] = True

                else:
                    done[i] = False

                    y = p.getContactPoints(self.PLANE_ID, self.DRONE_IDS[0])
                    if y != ():
                            done[i] = True
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

        MAX_X = 65
        MAX_Y = 15
        MAX_Z = 15

        MIN_X = -25
        MIN_Y = -15
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

        normalized_pos_x = self._normalize(clipped_pos_x, MIN_X, MAX_X)
        normalized_pos_y = self._normalize(clipped_pos_y, MIN_Y, MAX_Y)
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
        
        #self._processDistances()
        #exit(0)
        
        #b = self.ANGELO_processDistances()
        #print("b")
        #print(b)
        
        #return [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
        
        global distances
        MAX_DELTA = 10
        MIN_DELTA = -10
        self._processDistances()
        clippedAndNormalizedDistanceComponents = np.array([])
        nearest10distances = distances[:10]
        #print("Coordinates of the 10 nearest spheres")
        #print(nearest10distances)
        # Sorto le 10 sfere in ordine di coordinata x
        nearest10distances = nearest10distances[nearest10distances[:, 0].argsort()]
        #print("Sorted by X")
        #print(nearest10distances)
        for nearestSphereIndex in range(0, len(nearest10distances)):
            if nearest10distances[nearestSphereIndex][0] <= 10:
                if nearest10distances[nearestSphereIndex][0] < 0:
                    #sfera dietro
                    deltaX = -1
                    deltaY = 1
                    deltaZ = 1
                else:
                    #sfera davanti
                    deltaX = self._normalize(np.clip(nearest10distances[nearestSphereIndex][1], MIN_DELTA, MAX_DELTA), MIN_DELTA, MAX_DELTA)
                    deltaY = self._normalize(np.clip(nearest10distances[nearestSphereIndex][2], MIN_DELTA, MAX_DELTA), MIN_DELTA, MAX_DELTA)
                    deltaZ = self._normalize(np.clip(nearest10distances[nearestSphereIndex][3], MIN_DELTA, MAX_DELTA), MIN_DELTA, MAX_DELTA)
            else:
                #sfera troppo lontana
                deltaX = -1
                deltaY = 1
                deltaZ = 1

            clippedAndNormalizedDistanceComponents = np.append(clippedAndNormalizedDistanceComponents, [deltaX, deltaY, deltaZ])

        #print(clippedAndNormalizedDistanceComponents)
        return clippedAndNormalizedDistanceComponents
    
     # Mappa un valore dall'intervallo min-max all'intervallo -1/1
    def _normalize(self, val, min, max):
        return np.float32((val-min)/(max-min))*2-1
    
    def _normalizeTwo(self, val, min, max):
	    return np.array([self._normalize(val[0],min[0],max[0]), self._normalize(val[1],min[1],max[1])])

    '''
    Mi aspetto che l'array delle posizioni delle sfere abbia shape (numSpheres, 4)
    Cioè sia fatto così [ [Xs1, Ys1, Zs1, Rs1], [Xs2, Ys2, Zs2, Rs2], ...  ] (R è radius)
    Restituisco un array con shape (numSpheres, 4)
    Cioè fatto così [ [∆x1, ∆y1, ∆z1, distanceNorm1], [∆x2, ∆y2, ∆z2, distanceNorm2], ...]
    Sorta i risultati per norma
    '''
    def _processDistances(self):  
        global distances
        global cached_spheres

        unsortedDistances = np.zeros(shape=(len(cached_spheres), 5))
        #for droneIndex in range(0, self.NUM_DRONES):
        droneIndex = 0
        stateVector = self._getDroneStateVector(droneIndex)
        for sphereIndex in range(0, len(cached_spheres)): 
            dronePositionVector = np.array(stateVector[0:3], dtype='float64')
            spherePositionVector = np.array(cached_spheres[sphereIndex][1:4], dtype='float64')
            sphereRadius = 0.03*np.array(cached_spheres[sphereIndex][4], dtype='float64')

            distanceComponents = np.absolute(spherePositionVector - dronePositionVector)
            distanceComponentsCubic = distanceComponents - np.array([sphereRadius, sphereRadius, sphereRadius])
                
            distanceFromCenterNorm = np.linalg.norm(distanceComponents)
            distanceFromSurfaceNorm = distanceFromCenterNorm - sphereRadius
            # X della sfera, componenti della distanza dalla superficie del "cubo" e distanza dalla superficie della sfera
            # le doppie (()) servono !

            #aggiunta differenza tra x della sfera e x del drone per avere la posizione relativa sull'asse
            #e rendere più immediato il controllo "sfera-davanti/dietro-drone"
            relativeX = spherePositionVector[0] - stateVector[0]

            singleDistance = np.concatenate(([relativeX], distanceComponentsCubic, [distanceFromSurfaceNorm]))
            unsortedDistances[sphereIndex] = singleDistance
            
            """
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
            """
        #print('Distances processed')
        #print(unsortedDistances)
        # Sort by the 4th element of the inner arrays (the norm)
        distances = unsortedDistances[unsortedDistances[:, 4].argsort()]
        #print('Distances sorted by distance from surface')
        #print(distances)

    def ANGELO_processDistances(self):
        global distances
        distances = np.array(cached_spheres)
        dataDelta = distances
        print(distances)
        print(dataDelta)
        droneIndex = 0
        dronePosition = self._getDroneStateVector(droneIndex)
        dataDelta[:,:3] = dataDelta[:,:3]-dronePosition-dataDelta[:,3:4]
        distancesT = np.array([np.linalg.norm(dataDelta[:,:3],axis=1)]).T
        dataDelta[:,3] = np.ravel(distancesT)
        return dataDelta[dataDelta[:,3].argsort()]
