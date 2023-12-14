import os
import json
import pandas as pd
from pathlib import Path
from cr_autophagy_pyo3 import *
import multiprocessing as mp
import numpy as np
from types import SimpleNamespace
import pyvista as pv
import matplotlib.pyplot as plt
import tqdm
import scipy as sp
import itertools
import cc3d
from dataclasses import dataclass


def get_last_output_path(name = "autophagy"):
    return Path("out") / name / sorted(os.listdir(Path("out") / name))[-1]


def get_simulation_settings(output_path):
    f = open(output_path / "simulation_settings.json")
    return json.load(f, object_hook=lambda d: SimpleNamespace(**d))


def _combine_batches(run_directory):
    # Opens all batches in a given directory and stores
    # them in one unified big list
    combined_batch = []
    for batch_file in os.listdir(run_directory):
        f = open(run_directory / batch_file)
        b = json.load(f)["data"]
        combined_batch.extend(b)
    return combined_batch


def get_particles_at_iter(output_path: Path, iteration):
    dir = Path(output_path) / "cell_storage/json"
    run_directory = None
    for x in os.listdir(dir):
        if int(x) == iteration:
            run_directory = dir / x
            break
    if run_directory != None:
        df = pd.json_normalize(_combine_batches(run_directory))
        df["identifier"] = df["identifier"].apply(lambda x: tuple(x))
        df["element.cell.mechanics.pos"] = df["element.cell.mechanics.pos"].apply(lambda x: np.array(x, dtype=float))
        df["element.cell.mechanics.vel"] = df["element.cell.mechanics.vel"].apply(lambda x: np.array(x, dtype=float))
        df["element.cell.mechanics.random_vector"] = df["element.cell.mechanics.random_vector"].apply(lambda x: np.array(x))
        return df
    else:
        raise ValueError(f"Could not find iteration {iteration} in saved results")


def get_all_iterations(output_path):
    return sorted([int(x) for x in os.listdir(Path(output_path) / "cell_storage/json")])


def __iter_to_cells(iteration_dir):
    iteration, dir = iteration_dir
    return (int(iteration), _combine_batches(dir / iteration))


def get_particles_at_all_iterations(output_path: Path, threads=1):
    dir = Path(output_path) / "cell_storage/json/"
    runs = [(x, dir) for x in os.listdir(dir)]
    pool = mp.Pool(threads)
    result = list(pool.map(__iter_to_cells, runs[:10]))
    return result


def generate_spheres(output_path: Path, iteration):
    # Filter for only particles at the specified iteration
    df = get_particles_at_iter(output_path, iteration)
    # df = df[df["iteration"]==iteration]

    # Create a dataset for pyvista for plotting
    pos_cargo = df[df["element.cell.interaction.species"]=="Cargo"]["element.cell.mechanics.pos"]
    pos_r11 = df[df["element.cell.interaction.species"]!="Cargo"]["element.cell.mechanics.pos"]
    pset_cargo = pv.PolyData(np.array([np.array(x) for x in pos_cargo]))
    pset_r11 = pv.PolyData(np.array([np.array(x) for x in pos_r11]))

    # Extend dataset by species and diameter
    pset_cargo.point_data["diameter"] = 2.0*df[df["element.cell.interaction.species"]=="Cargo"]["element.cell.interaction.cell_radius"]
    pset_cargo.point_data["species"] = df[df["element.cell.interaction.species"]=="Cargo"]["element.cell.interaction.species"]
    pset_cargo.point_data["neighbour_count1"] = df[df["element.cell.interaction.species"]=="Cargo"]["element.cell.interaction.neighbour_count"]

    pset_r11.point_data["diameter"] = 2.0*df[df["element.cell.interaction.species"]!="Cargo"]["element.cell.interaction.cell_radius"]
    pset_r11.point_data["species"] = df[df["element.cell.interaction.species"]!="Cargo"]["element.cell.interaction.species"]
    pset_r11.point_data["neighbour_count2"] = df[df["element.cell.interaction.species"]!="Cargo"]["element.cell.interaction.neighbour_count"]

    # Create spheres glyphs from dataset
    sphere = pv.Sphere()
    spheres_cargo = pset_cargo.glyph(geom=sphere, scale="diameter", orient=False)
    spheres_r11 = pset_r11.glyph(geom=sphere, scale="diameter", orient=False)

    return spheres_cargo, spheres_r11


def create_save_path(output_path, subfolder, iteration):
    # Create folder if not exists
    ofolder = Path(output_path) / subfolder
    ofolder.mkdir(parents=True, exist_ok=True)
    save_path = ofolder / "snapshot_{:08}.png".format(iteration)
    return save_path


def save_snapshot(output_path: Path, iteration, overwrite=False):
    simulation_settings = get_simulation_settings(output_path)
    opath = create_save_path(output_path, "snapshots", iteration)
    if os.path.isfile(opath) and not overwrite:
        return
    (cargo, r11) = generate_spheres(output_path, iteration)

    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            jupyter_backend = 'none'
    except:
        jupyter_backend = None

    # Now display all information
    plotter = pv.Plotter(off_screen=True)
    ds = 1.5*simulation_settings.domain_size
    plotter.camera_position = [
        (-ds, -ds, -ds),
        (ds, ds, ds),
        (0, 0, 0)
    ]

    scalar_bar_args1=dict(
        title="Neighbours",
        title_font_size=20,
        width=0.4,
        position_x=0.55,
        label_font_size=16,
        shadow=True,
        italic=True,
        fmt="%.0f",
        font_family="arial",
    )

    plotter.add_mesh(
        cargo,
        scalars="neighbour_count1",
        cmap="Blues",
        clim=[0,12],
        scalar_bar_args=scalar_bar_args1,
    )
    plotter.add_mesh(
        r11,
        scalars="neighbour_count2",
        cmap="Oranges",
        clim=[0,12],
        scalar_bar_args=scalar_bar_args1,
    )
    img = plotter.screenshot(opath)
    plotter.close()
    return img


def __save_snapshot_helper(args):
    return save_snapshot(*args)


def save_all_snapshots(output_path: Path, threads=1, show_bar=True):
    if threads<=0:
        threads = os.cpu_count()
    output_iterations = [(output_path, iteration) for iteration in get_all_iterations(output_path)]
    if show_bar:
        list(tqdm.tqdm(mp.Pool(threads).imap(
            __save_snapshot_helper,
            output_iterations
        ), total=len(output_iterations)))
    else:
        mp.Pool(threads).imap(__save_snapshot_helper, output_iterations)


def save_scatter_snapshot(output_path: Path, iteration):
    df = get_particles_at_iter(output_path, iteration)

    cargo_at_end = df[df["element.cell.interaction.species"]=="Cargo"]["element.cell.mechanics.pos"]
    cargo_at_end = np.array([np.array(elem) for elem in cargo_at_end])
    non_cargo_at_end = df[df["element.cell.interaction.species"]!="Cargo"]["element.cell.mechanics.pos"]
    non_cargo_at_end = np.array([np.array(elem) for elem in non_cargo_at_end])
    cargo_middle = np.average(non_cargo_at_end, axis=0)

    def appendSpherical_np(xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
        ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
        return ptsnew

    non_cargo_at_end_spherical = appendSpherical_np(non_cargo_at_end - cargo_middle)
    r = non_cargo_at_end_spherical[:,3]
    r_inv = np.max(r) - r
    phi = non_cargo_at_end_spherical[:,4]
    theta = non_cargo_at_end_spherical[:,5]

    fig, ax = plt.subplots()
    ax.set_title("Radial distribution of particles around cargo center")
    ax.scatter(phi, theta, s=r_inv, alpha=0.5)

    ax.set_xlabel("$\\varphi$ [rad]")
    ax.set_ylabel("$\\theta$ [rad]")
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(["$0$", "$\\frac{\\pi}{4}$", "$\\frac{\\pi}{2}$", "$\\frac{3\\pi}{4}$", "$\\pi$"])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels(["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])

    ax.set_xlim([-np.pi/12, np.pi*(1+1/12)])
    ax.set_ylim([-np.pi*(1+1/6), np.pi*(1+1/6)])

    ofolder = output_path / "scatterplots"
    ofolder.mkdir(parents=True, exist_ok=True)
    fig.savefig(ofolder / f"snapshot_{iteration:08}_scatter.png")
    plt.close(fig)


def __save_scatter_snapshot_helper(args):
    return save_scatter_snapshot(*args)


def save_all_scatter_snapshots(output_path: Path, threads=1, show_bar=True):
    if threads<=0:
        threads = os.cpu_count()
    output_iterations = [(output_path, iteration) for iteration in get_all_iterations(output_path)]
    if show_bar:
        list(tqdm.tqdm(mp.Pool(threads).imap(
            __save_scatter_snapshot_helper,
            output_iterations
        ), total=len(output_iterations)))
    else:
        mp.Pool(threads).map(__save_scatter_snapshot_helper, output_iterations)


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


def save_cluster_information_plots(output_path, iteration, connection_distance=2.0, overwrite=False):
    opath = create_save_path(output_path, "clusterplots", iteration)

    # Get particles at specified iteration
    df = get_particles_at_iter(output_path, iteration)

    cargo_positions = df[df["element.cell.interaction.species"]=="Cargo"]["element.cell.mechanics.pos"]
    cargo_positions = np.array([np.array(elem) for elem in cargo_positions])
    non_cargo_positions = df[df["element.cell.interaction.species"]!="Cargo"]["element.cell.mechanics.pos"]
    non_cargo_positions = np.array([np.array(elem) for elem in non_cargo_positions])

    # Set max distance at which two cells are considered part of same cluster
    cargo_center = np.average(cargo_positions, axis=0)

    n_components, cluster_positions, cluster_sizes, min_cluster_distances_to_cargo = calculate_clusters(
        positions=non_cargo_positions,
        distance=connection_distance,
        cargo_position=cargo_center,
    )

    def calculate_cargo_percentile_boundary(
        positions: np.ndarray,
        center: np.ndarray,
        percentile: float,
        percentile_upper: float = 100,
    ):
        particles_distance = np.array([np.sqrt(np.sum((x-center)**2)) for x in positions])
        percentile_low = np.percentile(particles_distance, percentile)
        percentile_high = np.percentile(particles_distance, percentile_upper)

        return [percentile_low, percentile_high]

    percentiles = [(0,70),(70, 80), (80, 90), (90, 100)]
    boundaries = [calculate_cargo_percentile_boundary(
        cargo_positions,
        cargo_center,
        x,
        y,
    ) for (x,y) in percentiles]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0,0].scatter(min_cluster_distances_to_cargo, cluster_sizes, color="#A23302")
    ax[0,0].set_xlabel('min distance of cluster to cargo')
    ax[0,0].set_ylabel('# particles in cluster')#
    for i, boundary in enumerate(boundaries):
        p = percentiles[i][1]
        ax[0,0].axvspan(*boundary, alpha=(100-p)/50, color="#072853")

    x_lim_high = 1.05*np.max(min_cluster_distances_to_cargo)
    x_lim_low = -0.05*x_lim_high
    ax[0,0].set_xlim([x_lim_low, x_lim_high])

    ax[0,1].hist(cluster_sizes, color="#A23302")
    ax[0,1].set_xlabel('# particles in cluster')
    ax[0,1].set_ylabel('# of clusters')

    count, bins = np.histogram(cluster_sizes)

    # finding the PDF of the histogram using count values 
    pdf = count / sum(count) 

    # using numpy np.cumsum to calculate the CDF 
    # We can also find using the PDF values by looping and adding 
    cdf = np.cumsum(pdf) 

    # plotting PDF and CDF 
    ax[1,1].plot(bins[1:], pdf, color="#A23302", label="PDF")
    ax[1,1].plot(bins[1:], cdf, label="CDF", color="#A23302", linestyle="--")
    ax[1,1].set_xlabel('# particles in cluster')
    ax[1,1].set_ylabel('prob. density / cum. prob. density')
    ax[1,1].legend()

    img = save_snapshot(output_path, iteration, overwrite=True)
    ax[1,0].imshow(img)
    ax[1,0].axis('off')

    fig.savefig(opath)
    return fig


def __save_cluster_information_plots_helper(args):
    save_cluster_information_plots(*args)


def save_all_cluster_information_plots(
    output_path: Path,
    threads=1,
    show_bar=True,
    connection_distance=2.0
):
    if threads<=0:
        threads = os.cpu_count()
    output_iterations = [
        (output_path, iteration,connection_distance)
        for iteration in get_all_iterations(output_path)
    ]
    if show_bar:
        list(tqdm.tqdm(mp.Pool(threads).imap(
            __save_cluster_information_plots_helper,
            output_iterations
        ), total=len(output_iterations)))
    else:
        mp.Pool(threads).map(__save_cluster_information_plots_helper, output_iterations)


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


def _calculate_mask(density, threshold):
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


def save_kernel_density(
    output_path,
    iteration,
    threshold=0.45,
    overwrite=False,
    discretization_factor=0.5,
    bw_method=None,
):
    save_path = create_save_path(output_path, "kernel_density", iteration)
    if os.path.isfile(save_path) and overwrite==False:
        return None

    density_cargo, density_r11 = calculate_kernel_densities(
        output_path,
        iteration,
        discretization_factor,
        bw_method
    )

    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    ax[0,0].set_title("R11 Density")
    ax[0,1].set_title("R11 Mask")
    ax[0,2].set_title("Cargo Density")
    ax[0,3].set_title("Cargo Mask")

    for i in range(3):
        lims_low = [0 if i!=j else round(density_r11.shape[0]/2) for j in range(3)]
        lims_high = [-1 if i!=j else round(density_r11.shape[0]/2)+1 for j in range(3)]

        data_r11 = density_r11[lims_low[0]:lims_high[0],lims_low[1]:lims_high[1],lims_low[2]:lims_high[2]]
        data_r11 = data_r11.squeeze(axis=i)
        mask_r11, _ = _calculate_mask(data_r11, threshold)

        data_cargo = density_cargo[lims_low[0]:lims_high[0],lims_low[1]:lims_high[1],lims_low[2]:lims_high[2]]
        data_cargo = data_cargo.squeeze(axis=i)
        mask_cargo, _ = _calculate_mask(data_cargo, threshold)

        # Plot density of R11
        ax[i,0].imshow(data_r11)
        ax[i,0].axis('off')

        # Plot mask of R11
        ax[i,1].imshow(mask_r11, cmap="grey")
        ax[i,1].axis('off')

        # Plot Density of Cargo
        ax[i,2].imshow(data_cargo)
        ax[i,2].axis('off')

        ax[i,3].imshow(mask_cargo, cmap="grey")
        ax[i,3].axis('off')
    fig.tight_layout()
    fig.savefig(save_path)
    return fig


def _save_kernel_density_helper(args_kwargs):
    args, kwargs = args_kwargs
    fig = save_kernel_density(*args, **kwargs)
    plt.close(fig)


def save_all_kernel_density(output_path, threads=1, **kwargs):
    if threads<=0:
        threads = mp.cpu_count()
    pool = mp.Pool(threads)
    args = [((output_path, iteration), kwargs) for iteration in get_all_iterations(output_path)]
    _ = list(tqdm.tqdm(pool.imap_unordered(_save_kernel_density_helper, args), total=len(args)))


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
        mask_cargo, _ = _calculate_mask(density_cargo, threshold)

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

    mask_r11,   _ = _calculate_mask(density_r11,   threshold)
    mask_cargo, _ = _calculate_mask(density_cargo, threshold)

    return calculate_cargo_r11_cluster_distances(mask_r11, mask_cargo, domain_size)


def plot_cluster_distribution(output_path, iteration, discretization_factor, bw_method):
    clrs = get_clusters_kde(output_path, iteration, discretization_factor, bw_method)

    # Calculate percentiles for plotting
    percentiles = [clrs.get_cargo_distance_percentile(perc) for perc in [70, 80, 90]]

    distances = np.array([
        np.sum((x-clrs.cargo_position)**2)**0.5 for x in clrs.cluster_positions
    ])

    fig, ax = plt.subplots()
    # ax.hist(labels[labels!=np.argmax(cluster_sizes)])
    ax.set_ylabel("Cluster Volume")
    ax.set_xlabel("Distance to Cargo")
    ax.set_title(f"{clrs.n_clusters-1} Cluster" + (clrs.n_clusters-1>1)*"s")
    ax.scatter(distances, clrs.cluster_sizes[1:], color="#A23302")

    last_percentile = 0
    max_percentile = np.max(percentiles)
    for p in percentiles:
        ax.axvspan(last_percentile, p, alpha=1-p/max_percentile, color="#072853")
        last_percentile = p
    return fig
