import numpy as np
import scipy as sp
import itertools
import multiprocessing as mp
import cc3d

from dataclasses import dataclass

# Internal imports
from .storage import *


def calculate_clusters(positions: np.ndarray, distance: float, cargo_position: np.ndarray):
    # Calculate the combined matrix of all positions
    combined_matrix = np.array(list(itertools.product(positions, positions)))
    combined_matrix = combined_matrix.reshape(
        (positions.shape[0], positions.shape[0], *combined_matrix.shape[1:])
    )

    # Calculate the distances between individual positions in two steps
    # First Calculate the difference and omit unneeded axis
    distance_matrix = np.squeeze(np.diff(combined_matrix, axis=2), axis=2)
    # Then calculate the square, sum and square-root again
    distance_matrix = np.sqrt(np.sum(distance_matrix**2, axis=2))

    # Check that all distances of elements to themselves are actually zero
    assert np.all(np.diag(distance_matrix) == 0)

    # Construct Connection Matrix by filtering for distances below the specified threshold
    connection_matrix = distance_matrix < distance

    # Make sure that we actually choose more values than just the diagonal ones
    assert np.sum(connection_matrix) > connection_matrix.shape[0]

    # Calculate all connected components via scipy
    n_components, labels = sp.sparse.csgraph.connected_components(
        connection_matrix,
        directed=False
    )

    # Calculate the sizes of clusters, meaning how many particles are in one cluster
    cluster_sizes = np.array([
        np.sum(labels==label) for label in np.sort(np.unique(labels))
    ])

    _helper = lambda x: np.sqrt(np.sum((x - cargo_position)**2))

    # Also calculate the minimum distance from cluster center to cargo center
    cluster_positions = np.array([
        np.average(positions[labels==label], axis=0) for label in np.sort(np.unique(labels))
    ])
    min_cluster_distances_to_cargo = np.array([
        np.min([_helper(x) for x in positions[labels==label]], axis=0)
        for label in np.sort(np.unique(labels))
    ])

    # Return all results
    return n_components, cluster_positions, cluster_sizes, min_cluster_distances_to_cargo


@dataclass
class GraphClusterResult:
    n_clusters: int
    cluster_positions: np.ndarray
    cluster_sizes: np.ndarray
    min_cluster_distances_to_cargo: np.ndarray
    cargo_center: np.ndarray
    cargo_distances: np.ndarray
    cargo_positions: np.ndarray


    def get_cargo_distance_percentile(self, percentile):
        return np.percentile(self.cargo_distances, percentile)


    def validate(self):
        return True


    def clusters_at_cargo(self, relative_radial_distance):
        # TODO fix this function
        cargo_radius = self.get_cargo_distance_percentile(90)
        cluster_cargo_distances = np.sum((self.cluster_positions-self.cargo_center)**2, axis=1)**0.5
        mask = cluster_cargo_distances < (1.0 + relative_radial_distance) * cargo_radius
        return self.cluster_positions[mask]


def get_clusters_graph(output_path, iteration, connection_distance=2.0):
    # Get particles at specified iteration
    df = get_particles_at_iter(output_path, iteration)

    cargo_positions = df[df["element.cell.interaction.species"]=="Cargo"]["element.cell.mechanics.pos"]
    cargo_positions = np.array([np.array(elem) for elem in cargo_positions])
    non_cargo_positions = df[df["element.cell.interaction.species"]!="Cargo"]["element.cell.mechanics.pos"]
    non_cargo_positions = np.array([np.array(elem) for elem in non_cargo_positions])

    # Set max distance at which two cells are considered part of same cluster
    cargo_center = np.average(cargo_positions, axis=0)
    cargo_distances = np.sum((cargo_positions-cargo_center)**2, axis=1)**0.5

    n_components, cluster_positions, cluster_sizes, min_cluster_distances_to_cargo = calculate_clusters(
        positions=non_cargo_positions,
        distance=connection_distance,
        cargo_position=cargo_center,
    )
    return GraphClusterResult(
        n_components,
        cluster_positions,
        cluster_sizes,
        min_cluster_distances_to_cargo,
        cargo_center,
        cargo_positions,
        cargo_distances,
    )



def calculate_spatial_discretization(domain_size, discretization):
    xmin = 0.0
    xmax = domain_size
    h = discretization

    X, Y, Z = np.mgrid[xmin:xmax:h, xmin:xmax:h, xmin:xmax:h]

    space_discretization = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    return X, Y, Z, space_discretization


def calculate_spatial_density(particle_positions, domain_size, discretization, bw_method=None):
    x = particle_positions[:,0]
    y = particle_positions[:,1]
    z = particle_positions[:,2]

    X, Y, Z, space_discretization = calculate_spatial_discretization(domain_size, discretization)

    values = np.vstack([x, y, z])
    kernel = sp.stats.gaussian_kde(values, bw_method)
    D = np.reshape(kernel(space_discretization).T, X.shape)
    return D


def calculate_mask(density, threshold):
    thresh = threshold*(np.max(density)  -np.min(density))   + np.min(density)
    mask = density>=thresh
    return mask, thresh


def _get_discretization(output_path, discretization_factor):
    simulation_settings = get_simulation_settings(output_path)
    radius_cargo = simulation_settings.cell_radius_cargo
    radius_r11 = simulation_settings.cell_radius_r11
    return discretization_factor*radius_cargo, discretization_factor*radius_r11


def calculate_kernel_densities(output_path, iteration, discretization_factor, bw_method):
    simulation_settings = get_simulation_settings(output_path)
    domain_size = simulation_settings.domain_size

    discr_cargo, discr_r11 = _get_discretization(output_path, discretization_factor)

    df_cells = get_particles_at_iter(output_path, iteration)
    positions_cargo = np.array([x
        for x in df_cells[
            df_cells["element.cell.interaction.species"]=="Cargo"]["element.cell.mechanics.pos"
        ]
    ])
    positions_r11 = np.array([x
        for x in df_cells[
            df_cells["element.cell.interaction.species"]!="Cargo"]["element.cell.mechanics.pos"
        ]
    ])

    density_cargo = calculate_spatial_density(
        positions_cargo,
        domain_size,
        discr_cargo,
        bw_method
    )
    density_r11 = calculate_spatial_density(
        positions_r11,
        domain_size,
        discr_r11,
        bw_method
    )
    return density_cargo, density_r11



def calcualte_3d_connected_components(mask):
    labels = cc3d.connected_components(mask)
    cluster_identifiers = np.unique(labels)
    n_clusters = len(cluster_identifiers)
    cluster_sizes = np.array([len(labels[labels==i]) for i in cluster_identifiers])
    return n_clusters, labels, cluster_identifiers, cluster_sizes


def determine_optimal_thresh(
        output_path,
        iteration,
        discretization_factor,
        bw_method=0.45,
        dthresh=0.01,
    ):
    # Define starting values
    n_clusters = 2
    threshold = dthresh

    # As long as the cargo is still recognized as one cluster, we are lowering
    while n_clusters == 2:
        threshold += dthresh

        density_cargo, _ = calculate_kernel_densities(
            output_path,
            iteration,
            discretization_factor,
            bw_method
        )
        mask_cargo, _ = calculate_mask(density_cargo, threshold)

        n_clusters, _, _, _ = calcualte_3d_connected_components(mask_cargo)

    # We return the last known value where bw_method was successful
    return threshold - dthresh


@dataclass
class KDEClusterResult:
    n_clusters: int
    cluster_sizes: np.ndarray
    cluster_positions: np.ndarray
    cargo_position: np.ndarray
    distances_cargo: np.ndarray

    def get_cargo_distance_percentile(self, percentile):
        return np.percentile(self.distances_cargo, percentile)


    def _validate_leakiness(self, percentile=70, **kwargs):
        cargo_distance = self.get_cargo_distance_percentile(percentile)
        return np.all(
            np.sum((self.cluster_positions-cargo_distance)**2, axis=1)**0.5>cargo_distance
        )


    def validate(self, **kwargs):
        if self._validate_leakiness(**kwargs) == False:
            return False
        return True


    def clusters_at_cargo(self, relative_radial_distance=0.5):
        cargo_radius = self.get_cargo_distance_percentile(90)
        cluster_cargo_distances = np.sum((self.cluster_positions-self.cargo_position)**2, axis=1)**0.5
        mask = cluster_cargo_distances < (1.0 + relative_radial_distance) * cargo_radius
        return self.cluster_positions[mask]


def calculate_cargo_r11_cluster_distances(mask_r11, mask_cargo, domain_size) -> KDEClusterResult:
    _, labels_cargo, identifiers_cargo, _ = calcualte_3d_connected_components(mask_cargo)
    n_clusters, labels_r11, identifiers_r11, cluster_sizes = calcualte_3d_connected_components(mask_r11)

    _helper_x = domain_size/np.array([*labels_cargo.shape])
    _helper_y = np.product(_helper_x)**(1/3)
    to_coordinate = lambda x: x*_helper_x
    to_coordinate_single = lambda x: x*_helper_y

    # Transform cluster_sizes to coordinates
    cluster_sizes = to_coordinate_single(cluster_sizes)

    positions_cargo = np.array([])
    for ident in identifiers_cargo[1:]:
        indices = np.argwhere(labels_cargo==ident)
        middle = to_coordinate(np.average(indices, axis=0))
        distances_cargo = np.sum((to_coordinate(indices) - middle)**2, axis=1)**0.5
        positions_cargo = np.vstack([*positions_cargo, middle])

    # Make sure that we only have one cargo position left over
    if positions_cargo.shape[0] != 1:
        return

    cargo_position = positions_cargo[0]
    if cargo_position.shape != (3,):
        return

    # Now gather positions of R11 clusters
    cluster_positions = np.array([])
    for ident in identifiers_r11[1:]:
        indices = np.argwhere(labels_r11==ident)
        middle = np.average(indices, axis=0)*domain_size/np.array([*labels_r11.shape])
        cluster_positions = np.vstack([*cluster_positions, middle])

    # Make sure that we really have some R11 clusters
    if cluster_positions.shape[0] == 0:
        return

    return KDEClusterResult(
        n_clusters-1,
        cluster_sizes[1:],
        cluster_positions[1:],
        cargo_position,
        distances_cargo
    )


def get_clusters_kde(output_path, iteration, *args):
    simulation_settings = get_simulation_settings(output_path)
    domain_size = simulation_settings.domain_size

    threshold = determine_optimal_thresh(
        output_path,
        iteration,
        *args,
    )

    density_cargo, density_r11 = calculate_kernel_densities(
        output_path,
        iteration,
        *args,
    )

    mask_r11,   _ = calculate_mask(density_r11,   threshold)
    mask_cargo, _ = calculate_mask(density_cargo, threshold)

    return calculate_cargo_r11_cluster_distances(mask_r11, mask_cargo, domain_size)
