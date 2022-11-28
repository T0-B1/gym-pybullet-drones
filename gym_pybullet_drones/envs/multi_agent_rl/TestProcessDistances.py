import numpy as np

MAX_DELTA_X = 10
MAX_DELTA_Y = 10
MAX_DELTA_Z = 10
#MAX_DIST = np.linalg.norm([MAX_DELTA_X, MAX_DELTA_Y, MAX_DELTA_Z])
MAX_DIST = 10

class ReachThePointAviarySingle():

    import numpy as np

    # shape(numSpheres, 5)
    # [ numSpheres, [prefab_name,pos_x,pos_y,pos_z,radius] ]
    cached_spheres = np.array([ 
        ['s1', -6, -5, -1, 5],
        ['s2', 1, 2, 3, 1],
        ['s3', 1, 1, 1, 1],
        ])
        
    distances = np.array([])
        
    def _getDroneStateVector(self, index):
        return [0,0,0]

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
            

        print('Distances processed')
        print(unsortedDistances)
        # Sort by the 4th element of the inner arrays (the norm)
        self.distances = unsortedDistances[unsortedDistances[:, 4].argsort()]
        print('Distances sorted by distance from surface')
        print(self.distances)
        

def main():
    rtpa = ReachThePointAviarySingle() 
    #rtpa._processDistances()
    clipped = rtpa._clipAndNormalizeSphere()
    print("Cubic distances clipped an normalized")
    print(clipped)


if __name__ == "__main__":
    main()
    