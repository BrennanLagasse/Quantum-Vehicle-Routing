# General components of the VRP problem

from dimod import (
    Binary,
    BinaryQuadraticModel,
    ConstrainedQuadraticModel,
)

from abc import ABC, abstractmethod

from enum import Enum

from dwave.system import LeapHybridCQMSampler

import math

class QuboForm(Enum):
    CLASSIC = 1
    PATH_EFFICIENT = 2
    PATH_EFFICIENT_BINARY = 3
    EDGE_MATRIX = 4


class VehicleRoutingModel(ABC):
    def __init__(self, num_locations: int, distances: list[list[int]], num_vehicles: int, max_distance: int):
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.distances = distances
        self.max_distance = max_distance
        self.obj = None
        self.cqm = ConstrainedQuadraticModel()

    def build_constrained_model(self):
        """Generates the constrained model for the VRP. The depot is given the zero ID"""
        print("Building model")
        self._construct_objective()
        self._add_constraints()

    def run_constrained_model(self, token):
        """Run the CQM on the DWave annealer. Requires a token."""
        
        sampler = LeapHybridCQMSampler(token=token)

        sampleset = sampler.sample_cqm(self.cqm)
        feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

        num_feasible = len(feasible_sampleset)
        errors = " "
        if num_feasible == 0:
            print("No feasible solution found.")
            return sampleset

        print("\nFeasible solution found.\n")

        return feasible_sampleset

    @abstractmethod
    def _construct_objective(self):
        """Generate the unconstrained objective function for the routing problem"""
        pass

    @abstractmethod
    def _add_constraints(self):
        """Add the constraints to the routing problem"""
        pass

class EdgeModel(VehicleRoutingModel):
    def __init__(self, num_locations: int, distances: list[list[int]], num_vehicles: int, max_distance: int):
        super().__init__(num_locations, distances, num_vehicles, max_distance)

    def construct_objective(self):

        M = self.num_vehicles
        N = self.num_locations

        # Create all the variables: one for each location/position combo
        # i is start vertex, j is end vertex
        x = {(i, j): Binary(str(i) + "_" + str(j)) for i in range(N+1) for j in range(N+1)}

        # Define the unconstrained binary optimization problem
        obj = BinaryQuadraticModel(vartype="BINARY")

        # The cost is just the sum over the edges
        for i in range(N+1):
            for j in range(N+1):
                obj += x[i,j] * self.distances[i,j]

        return x, obj
    
    def run_constrained_model(self):
        raise NotImplementedError

class DefaultRoutingModel(VehicleRoutingModel):

    def __init__(self, num_locations: int, distances: list[list[int]], num_vehicles: int, max_distance: int):
        super().__init__(num_locations, distances, num_vehicles, max_distance)

    def _construct_objective(self):

        self.obj = None

        M = self.num_vehicles
        N = self.num_locations

        # Create all the variables: one for each vehicle/location/position combo
        # k is timestep, j is vertex, i is vehicle
        self.x = {(i, j, k): Binary(str(i) + "_" + str(j) + "_" + str(k)) for k in range(N) for j in range(N+1) for i in range(M)}

        # Define the unconstrained binary optimization problem
        self.obj = BinaryQuadraticModel(vartype="BINARY")

        # The cost of going from the depot to the first stop
        for m in range(M):
            for n in range(1, N+1):
                self.obj += self.x[m, n, 0] * self.distances[0][n]

        # The cost of going from the last stop to the depot
        for m in range(M):
            for n in range(1, N+1):
                self.obj += self.x[m, n, N-1] * self.distances[n][0]

        # The cost of going between all stops in the middle
        for m in range(M):
            for t in range(N-1):
                for i in range(N+1):
                    for j in range(N+1):
                        self.obj += self.x[m, i, t] * self.x[m, j, t + 1] * self.distances[i][j]

        self.cqm.set_objective(self.obj)
    
    def _add_constraints(self):

        print("Adding constraints")

        M = self.num_vehicles
        N = self.num_locations

        # 1. Each location should be served by exactly one vehicle (does not check first since depot is required start and end location by construction)
        for j in range(1, N+1):
            sum = 0
            for m in range(M):
                for t in range(N):
                    sum += self.x[m, j, t]

            # Could be relaxed to geq, the rest should be handled by the objective
            self.cqm.add_constraint(sum == 1,
                            label=f"Vertex {j} is not visited or visited more than once")

        # 2. Each vehicle is in one location
        for i in range(M):
            for t in range(N):
                sum = 0
                for j in range(N + 1):
                    sum += self.x[i, j, t]
                self.cqm.add_constraint(sum == 1,
                                label=f"Vehicle {i} is at more or less than one position at time {t}")
                
        #3. Each vehicle drives less than the cap
        for m in range(M):
            sum = 0

            for i in range(N+1):
                
                # Add in the distances from deport to first, last to depot
                sum += self.x[m, i, 0]*self.distances[0][i]
                sum += self.x[m, i, N-1]*self.distances[i][0]

                # Go through the steps in the middle
                for j in range(N+1):
                    for t in range(N-1):
                        sum += self.x[m, i, t]*self.x[m, j, t+1]*self.distances[i][j]

            self.cqm.add_constraint(sum <= self.max_distance,
                                label=f"Vehicle {m} drives more than the maximum capacity")
            
class BoundedPathModel(VehicleRoutingModel):

    def __init__(self, num_locations: int, distances: list[list[int]], num_vehicles: int, max_distance: int):
        super().__init__(num_locations, distances, num_vehicles, max_distance)
        self.path_lengths = [math.ceil(num_locations / (m + 1)) for m in range(num_vehicles)]

    def _construct_objective(self):

        print("In bounded path objective")

        self.obj = None

        M = self.num_vehicles
        N = self.num_locations

        # Create all the variables: one for each vehicle/location/position combo
        # (i=vehicle, j=vertex, k=timestep)
        self.x = {(i, j, k): Binary(str(i) + "_" + str(j) + "_" + str(k)) for i in range(M) for j in range(N+1) for k in range(self.path_lengths[i])}

        # Define the unconstrained binary optimization problem
        self.obj = BinaryQuadraticModel(vartype="BINARY")

        # The cost of going from the depot to the first stop (unchanged)
        for m in range(M):
            for n in range(1, N+1):
                self.obj += self.x[m, n, 0] * self.distances[0][n]

        # The cost of going from the last stop to the depot (changed given longest path lengths)
        for m in range(M):
            for n in range(1, N+1):
                self.obj += self.x[m, n, self.path_lengths[m]-1] * self.distances[n][0]

        # The cost of going between all stops in the middle (changed given longest path lengths)
        for m in range(M):
            for t in range(self.path_lengths[m]-1):
                for i in range(N+1):
                    for j in range(N+1):
                        self.obj += self.x[m, i, t] * self.x[m, j, t + 1] * self.distances[i][j]

        self.cqm.set_objective(self.obj)
    
    def _add_constraints(self):

        print("In bounded path constraints")

        M = self.num_vehicles
        N = self.num_locations

        # 1. Each location should be served by exactly one vehicle (does not check first since depot is required start and end location by construction)
        for j in range(1, N+1):
            sum = 0
            for m in range(M):
                for t in range(self.path_lengths[m]):
                    sum += self.x[m, j, t]

            # Could be relaxed to geq, the rest should be handled by the objective
            self.cqm.add_constraint(sum == 1,
                            label=f"Vertex {j} is not visited or visited more than once")

        # 2. Each vehicle is in one location
        for m in range(M):
            for t in range(self.path_lengths[m]):
                print(f"adding pair {m}, {t}")
                sum = 0
                for j in range(N + 1):
                    sum += self.x[m, j, t]
                self.cqm.add_constraint(sum == 1,
                                label=f"Vehicle {m} is at more or less than one position at time {t}")
                
        #3. Each vehicle drives less than the cap
        for m in range(M):
            sum = 0

            for i in range(N+1):
                
                # Add in the distances from deport to first, last to depot
                sum += self.x[m, i, 0]*self.distances[0][i]
                sum += self.x[m, i, self.path_lengths[m]-1]*self.distances[i][0]

                # Go through the steps in the middle
                for j in range(N+1):
                    for t in range(self.path_lengths[m]-1):
                        sum += self.x[m, i, t]*self.x[m, j, t+1]*self.distances[i][j]

            self.cqm.add_constraint(sum <= self.max_distance,
                                label=f"Vehicle {m} drives more than the maximum capacity")

# TODO: Rewrite to use an appropriate higher-order solver
class NaiveBinaryModel(VehicleRoutingModel):
    
    def __init__(self, num_locations: int, distances: list[list[int]], num_vehicles: int, max_distance: int):
        super().__init__(num_locations, distances, num_vehicles, max_distance)

    def get_distance(self, i, j):
        """Pad the distance array to decentivise selecting invalid distances, avoid array miss"""
        if i <= self.num_locations and j <= self.num_locations:
            return self.distance[i][j]
        else:
            # Arbitrary high weight so that another route is incentivised
            return self.max_distance * 2

    def _check_location(self, vehicle, location, time):
        """Return the binary expression to check if a vehicle is at a location at a time"""

        term = 1
        indices = [int(bit) for bit in bin(location)[2:]]

        for idx, val in enumerate(indices):
            if val == 0:
                term *= (1-self.x[vehicle, idx, time])
            else:
                term *= self.x[vehicle, idx, time]

        return term


    def _construct_objective(self):

        self.obj = None

        M = self.num_vehicles
        N = self.num_locations
        N2 = 2**math.ceil(math.log(N+1, 2))

        # Create all the variables: one for each vehicle/location/position combo
        # (i=vehicle, j=vertex (one of the indices), k=timestep)
        self.x = {(i, j, k): Binary(str(i) + "_" + str(j) + "_" + str(k)) for k in range(N) for j in range(math.ceil(math.log(N+1, 2))) for i in range(M)}

        print(self.x.keys())

        # Define the unconstrained binary optimization problem
        self.obj = BinaryQuadraticModel(vartype="BINARY")

        # The cost of going from the depot to the first stop
        for m in range(M):
            for n in range(1, N2):
                self.obj += self._check_location(m, n, 0) * self.distances[0][n]

        # The cost of going from the last stop to the depot
        for m in range(M):
            for n in range(1, N2):
                self.obj += self._check_location(m, n, N-1) * self.distances[n][0]

        # The cost of going between all stops in the middle
        for m in range(M):
            for t in range(N+1):
                for i in range(N2):
                    for j in range(N2):
                        self.obj += self._check_location(m, i, t) * self._check_location(m, j, t+1) * self.distances[i][j]

        self.cqm.set_objective(self.obj)
    
    def _add_constraints(self):

        print("Adding constraints")

        M = self.num_vehicles
        N = self.num_locations

        # 1. Each location should be served by exactly one vehicle (does not check first since depot is required start and end location by construction)
        for j in range(1, N+1):
            sum = 0
            for m in range(M):
                for t in range(N):
                    sum += self._check_location(m, j, t)

            # Could be relaxed to geq, the rest should be handled by the objective
            self.cqm.add_constraint(sum == 1,
                            label=f"Vertex {j} is not visited or visited more than once")

        # 2. Each vehicle is in one location - unneccessary for this variant
                
        #3. Each vehicle drives less than the cap
        for m in range(M):
            sum = 0

            for i in range(N+1):
                
                # Add in the distances from deport to first, last to depot
                sum += self._check_location(m, i, 0)*self.get_distance(0, i)
                sum += self._check_location(m, i, N-1)*self.get_distance(i, 0)

                # Go through the steps in the middle
                for j in range(N+1):
                    for t in range(N-1):
                        sum += self._check_location(m, i, t)*self._check_location(m, j, t+1)*self.get_distance(i,j)

            self.cqm.add_constraint(sum <= self.max_distance,
                                label=f"Vehicle {m} drives more than the maximum capacity")