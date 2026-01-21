import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from frame.utils import prepare_graph_inputs
from frame.spatial_digraph import SpatialDiGraph
from frame.visualization import Vis
from frame.cross_validation import run_cv
from frame.digraphstats import Digraphstats

"""To get the genotype file, please download data from https://datadryad.org/dataset/doi:10.5061/dryad.7s848 and process the data using the procedures in https://github.com/elundgre/gene-flow-inference/tree/master/poplars"""

"""Please use poplars.coord and grid_220"""

outer, edges, grid, _ = prepare_graph_inputs(coord=coord,
                                             ggrid=grid_path,
                                             buffer=1)

grid_delete=[11,16,17,22,28,34,40,47,53]
mask = np.ones(len(grid), dtype=bool)
mask[grid_delete] = False
grid_new = grid[mask]

# Creating a mapping of old indexes to new indexes
mapping = {old_idx: new_idx for new_idx, old_idx in enumerate([i for i in range(len(grid)) if i not in grid_delete])}

valid_edges = np.array([edge for edge in edges if all(node in mapping for node in edge)])
# Adjusting edges based on new indexes

edges_new = np.vectorize(mapping.get)(valid_edges)

sp_digraph = SpatialDiGraph(genotypes, coord, grid_new, edges_new)

lamb_m_grid = np.geomspace(1e-3, 1e3,13)[::-1]
lamb_m_warmup =1e3

fr=1e10

cv_errs,node_train_idxs=run_cv(sp_digraph,
                               n_folds=10,
                               lamb_m_grid=lamb_m_grid,
                               lamb_m_warmup=lamb_m_warmup,
                               factr=fr,
                               random_state=500)

if np.argmin(cv_errs)==0:
   lamb_m_grid_fine=np.geomspace(lamb_m_grid[0],lamb_m_grid[1],7)[::-1]

elif np.argmin(cv_errs)==12:
     lamb_m_grid_fine=np.geomspace(lamb_m_grid[11],lamb_m_grid[12], 7)[::-1]
     
else:
    lamb_m_grid_fine=np.geomspace(lamb_m_grid[np.argmin(cv_errs)-1],lamb_m_grid[np.argmin(cv_errs)+1], 7)[::-1]

cv_errs_fine,node_train_idxs_fine=run_cv(sp_digraph,
                                         n_folds=10,
                                         lamb_m_grid=lamb_m_grid_fine,
                                         lamb_m_warmup=lamb_m_warmup,
                                         factr=fr,
                                         random_state=500,
                                         outer_verbose=True,
                                         inner_verbose=False,)

lamb_m_opt=lamb_m_grid_fine[np.argmin(cv_errs_fine)]
lamb_m_opt=float("{:.3g}".format(lamb_m_opt))

sp_digraph.fit(lamb_m=lamb_m_warmup, factr=fr)
logm = np.log(sp_digraph.m)
logc=np.log(sp_digraph.c)
trans_alpha=-np.log((1/sp_digraph.alpha)-1)

sp_digraph.fit(lamb_m=lamb_m_opt,
               factr=1e7,
               logm_init=logm,
               logc_init=logc,
               trans_alpha_init=trans_alpha,
               )

projection = ccrs.EquidistantConic(central_longitude=-108.842926, central_latitude=66.037547)

fig, axs= plt.subplots(2, 4, figsize=(8, 6), dpi=300,
                        subplot_kw={'projection': projection})

v = Vis(axs[0,0], sp_digraph, projection=projection, edge_width=1,
        edge_alpha=1, edge_zorder=100, sample_pt_size=10,
        obs_node_size=2, sample_pt_color="black",
        cbar_font_size=5, cbar_ticklabelsize=5,
        cbar_width="30%",
        cbar_height="3.5%",
        cbar_bbox_to_anchor=(0.05, 0.15, 1, 1), compass_bbox_to_anchor=(0, 0.075),
        compass_font_size=5,
        compass_radius=0.2,
        mutation_scale=6)

v.digraph_wrapper(axs, node_scale=[5, 5,10])

plt.figure(figsize=(8, 6))
plt.plot(np.log10(lamb_m_grid), cv, 'bo')  
plt.plot(np.log10(lamb_m_grid_fine), cv_fine, 'bo')  
plt.xlabel(r"$\mathrm{log}_{10}(\mathrm{\lambda_m})$")
plt.ylabel('CV Error')

digraphstats = Digraphstats(sp_digraph)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
digraphstats.distance_regression(ax,labelsize=8,R2_fontsize=12,legend_fontsize=12)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
digraphstats.z_score_distribution(ax)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
digraphstats.draw_heatmap(ax)
plt.show()
