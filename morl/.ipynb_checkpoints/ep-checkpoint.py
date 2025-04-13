import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from sample import Sample
from utils import get_ep_indices

'''
Define a external pareto class storing all computed policies on the current pareto front.
'''
class EP:
    def __init__(self, reference_point, num_select_solutions, policy_buffer_size):
        # Store pareto optimal solutions
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])
        
        # Store historical selected objective values to avoid repeated selection
        self.obj_hist = []
        
        # Store selected solutions for next extension iteration
        self.selected_batch = np.array([])
        self.selected_obj_batch = np.array([])
        
        self.policy_buffer_size = policy_buffer_size  # Maximum allowed size for sample_batch
        self.reference_point = reference_point
        self.num_select_solutions = num_select_solutions

    def crowding_distance_index(self, indices, inplace=True):
        self.selected_obj_batch = self.obj_batch[indices]
        self.selected_batch = self.sample_batch[indices]
        
    def index(self, indices, inplace=True):
        if inplace:
            self.obj_batch, self.sample_batch = \
                map(lambda batch: batch[np.array(indices, dtype=int)], [self.obj_batch, self.sample_batch])
        else:
            return map(lambda batch: deepcopy(batch[np.array(indices, dtype=int)]), [self.obj_batch, self.sample_batch])

    def update(self, sample_batch):
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))
        for sample in sample_batch:
            self.obj_batch = np.vstack([self.obj_batch, sample.objs]) if len(self.obj_batch) > 0 else np.array([sample.objs])
        if len(self.obj_batch) == 0: return
    
        ep_indices = get_ep_indices(self.obj_batch, self.reference_point)
        self.index(ep_indices)
        if len(self.sample_batch) > self.policy_buffer_size:
            print(f"Sample batch size exceeded policy buffer size {self.policy_buffer_size}. Performing sampling.")
            crowding_distances = self.calculate_crowding_distance(self.obj_batch, True)
            sorted_indices = np.argsort(-crowding_distances)  # Get indices sorted by crowding distance
            pareto_index = []
            for idx in sorted_indices:
                if len(pareto_index) < self.policy_buffer_size:
                    pareto_index.append(idx)
            self.sample_batch = self.sample_batch[pareto_index]
            self.obj_batch = self.obj_batch[pareto_index]
        print(f"Number of Pareto optimal solutions: {len(self.obj_batch)}")

        self.filter_by_crowding_distance(self.num_select_solutions) #number of elite policies
    

    def calculate_crowding_distance(self, obj_batch, extreme):
        if len(obj_batch) == 0:
            return np.array([])
        
        num_samples = obj_batch.shape[0]
        num_objectives = obj_batch.shape[1]
        crowding_distances = np.zeros(num_samples)

        for i in range(num_objectives):
            obj_values = obj_batch[:, i]
            sorted_indices = np.argsort(obj_values)
            distances = np.zeros(num_samples)
            distances[1:-1] = (obj_values[sorted_indices[2:]] - obj_values[sorted_indices[:-2]]) / \
                              (obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]])
            
            # Set the boundary points to infinity or a large number to ensure they are selected
            if extreme:
                distances[0] = distances[-1] = np.inf
            crowding_distances[sorted_indices] += distances

        return crowding_distances

    def filter_by_crowding_distance(self, num_select_solutions):
        crowding_distances = self.calculate_crowding_distance(self.obj_batch, True)
        sorted_indices = np.argsort(-crowding_distances)  # Get indices sorted by crowding distance

        # Select the top num_select_solutions policies, ensuring we don't select duplicates based on objective values
        new_indices = []
        for idx in sorted_indices:
            # Convert objectives to a tuple to use as a hashable type for comparison
            obj_tuple = tuple(self.obj_batch[idx])

            if obj_tuple not in self.obj_hist and len(new_indices) < num_select_solutions:
                new_indices.append(idx)
                self.obj_hist.append(obj_tuple)

        # Filter samples by these new indices using the existing self.index method
        self.crowding_distance_index(new_indices)
        
    def random_selection(self, num_select_solutions):
        """Randomly select num_select_solutions policies without considering crowding distance."""
        if len(self.obj_batch) < num_select_solutions:
            num_select_solutions = len(self.obj_batch)
        
        random_indices = np.random.choice(len(self.obj_batch), size=num_select_solutions, replace=False)
        self.index(random_indices)
