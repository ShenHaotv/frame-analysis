import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from frame.spatial_digraph import SpatialDiGraph
from frame.visualization import Vis
from frame.cross_validation import run_cv
from frame.digraphstats import Digraphstats

"""To get the genotype file, please download the v62.0_1240k_public dataset from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FFIDCW, apply the filters we showed in the supplementary, delete the repeating samples (for this step, we suggest using individual_list_neolithic.csv), and do the mean value imputation """

"""For the coord, grid, edges file, please use 7000-3500 BCE.coord, 7000-3500 BCE_grid.csv and 7000-3500 BCE_edges.csv"""

sp_digraph = SpatialDiGraph(genotypes, coord, grid, edges)

lamb_warmup = 1e3

lamb_grid = np.geomspace(1e-3,1e3,13)[::-1]

cv,node_train_idxs=run_cv(sp_digraph,
                          lamb_grid,
                          lamb_warmup=lamb_warmup,
                          n_folds=20,
                          factr=1e10,
                          random_state=500,
                          outer_verbose=True,
                          inner_verbose=False,)

if np.argmin(cv)==0:
   lamb_grid_fine=np.geomspace(lamb_grid[0],lamb_grid[1],7)[::-1]

elif np.argmin(cv)==12:
     lamb_grid_fine=np.geomspace(lamb_grid[11],lamb_grid[12], 12)[::-1]
     
else:
    lamb_grid_fine=np.geomspace(lamb_grid[np.argmin(cv)-1],lamb_grid[np.argmin(cv)+1], 7)[::-1]

cv_fine,node_train_idxs_fine=run_cv(sp_digraph,
                                    lamb_grid_fine,
                                    lamb_warmup=lamb_warmup,
                                    n_folds=20,
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

fig, axs= plt.subplots(2, 4, figsize=(15,6), dpi=300,
                        subplot_kw={'projection': projection})

v = Vis(axs[0,0], sp_digraph, projection=projection, edge_width=1,
        edge_alpha=1, edge_zorder=100, sample_pt_size=5,
        obs_node_size=0.5, sample_pt_color="black",
        cbar_font_size=5,cbar_ticklabelsize=8, 
        cbar_bbox_to_anchor=(0.4, -0.3, 1, 1),   
        campass_bbox_to_anchor=(0.4, -0.4),
        campass_font_size=5,
        campass_radius=0.2,
        mutation_scale=6)

v.digraph_wrapper(axs, node_scale=[5, 5, 5])
plt.subplots_adjust(hspace=0.1)

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


