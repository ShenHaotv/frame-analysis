import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from frame.utils import prepare_graph_inputs
from frame.spatial_digraph import SpatialDiGraph
from frame.visualization import Vis
from frame.cross_validation import run_cv
from frame.digraphstats import Digraphstats

"""To get the genotype file, please download 1960_SNPs.str file from the dataset from https://data.mendeley.com/datasets/w2tykkj77w/1, dataset 1, and do the mean value imputation"""

"""Please use manakin.coord, grid_440 and manakin.outer"""

outer, edges, grid, _ = prepare_graph_inputs(coord=coord,
                                             ggrid=grid_path,
                                             buffer=0,
                                             outer=outer)

grid[3,0]+=1
grid[69,0]+=0.5
grid[75,1]+=1
grid[76,:]=grid[76,:]+[-1.5,1.5]

sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)

lamb_warmup = 1e3

lamb_grid = np.geomspace(1e-3, 1e3,13)[::-1]

cv,node_train_idxs=run_cv(sp_digraph,
                          lamb_grid,
                          lamb_warmup=lamb_warmup,
                          n_folds=10,
                          factr=1e10,
                          random_state=500,
                          outer_verbose=True,
                          inner_verbose=False,)

if np.argmin(cv)==0:
   lamb_grid_fine=np.geomspace(lamb_grid[0],lamb_grid[1],7)[::-1]

elif np.argmin(cv)==12:
     lamb_grid_fine=np.geomspace(lamb_grid[11],lamb_grid[12], 7)[::-1]
     
else:
    lamb_grid_fine=np.geomspace(lamb_grid[np.argmin(cv)-1],lamb_grid[np.argmin(cv)+1], 7)[::-1]

cv_fine,node_train_idxs_fine=run_cv(sp_digraph,
                                    lamb_grid_fine,
                                    lamb_warmup=lamb_warmup,
                                    n_folds=10,
                                    factr=1e10,
                                    random_state=500,
                                    outer_verbose=True,
                                    inner_verbose=False,
                                    node_train_idxs=node_train_idxs)

lamb_opt=lamb_grid_fine[np.argmin(cv_fine)]
lamb_opt=float("{:.3g}".format(lamb_opt))

sp_digraph.fit(lamb=lamb_warmup, factr=1e10)
logm = np.log(sp_digraph.m)
logc = np.log(sp_digraph.c)

sp_digraph.fit(lamb=lamb_opt,
               factr=1e7,
               logm_init=logm,
               logc_init=logc,
               )


projection = ccrs.Mercator()
fig, axs= plt.subplots(2, 4, figsize=(16, 6), dpi=300,
                        subplot_kw={'projection': projection})

v = Vis(axs[0,0], sp_digraph, projection=projection, edge_width=1,
        edge_alpha=1, edge_zorder=100, sample_pt_size=10,
        obs_node_size=2, sample_pt_color="black",
        cbar_font_size=5, cbar_ticklabelsize=5,
        cbar_width="15%",
        cbar_height="3.5%",
        cbar_bbox_to_anchor=(0.05, 0.15, 1, 1), campass_bbox_to_anchor=(0,0.025),
        campass_font_size=5,
        campass_radius=0.15,
        mutation_scale=6)

v.digraph_wrapper(axs, node_scale=[5, 5, 5])


plt.subplots_adjust(hspace=0.2)

plt.figure(figsize=(8, 6))
plt.plot(np.log10(lamb_grid), cv, 'bo')  
plt.plot(np.log10(lamb_grid_fine), cv_fine, 'bo')  
plt.xlabel(r"$\mathrm{log}_{10}(\mathrm{\lambda})$")
plt.ylabel('CV Error')

digraphstats = Digraphstats(sp_digraph)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
digraphstats.distance_regression(ax,labelsize=8,R2_fontsize=12,legend_fontsize=12)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
digraphstats.z_score_distribution(ax)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
digraphstats.draw_heatmap(ax)
plt.show()
