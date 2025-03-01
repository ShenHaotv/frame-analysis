import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from .utils import prepare_graph_inputs
from .spatial_digraph import SpatialDiGraph
from .visualization import Vis
from .cross_validation import run_cv
from .digraphstats import Digraphstats

"""To get the genotype file, please download the v62.0_1240k_public dataset from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FFIDCW, apply the filters we showed in the supplementary, delete the repeating samples (for this step, we suggest using the neolithic_individual_list), and do the mean value imputation """

"""For the coord file, please use neolithic.coord, for the grid_path, please use grid_440"""

outer, edges, grid, _ = prepare_graph_inputs(coord=coord,
                                             ggrid=grid_path,
                                             buffer=2,)

grid[0,:]=grid[0,:]+[-1,-1]
grid[1,0]-=1
grid[2,:]=grid[2,:]+[2,-1]
grid[4,0]+=1
grid[6,0]+=1
grid[7,0]+=2
grid[8,:]=grid[8,:]+[4.5,-1.5]
grid[9,:]=grid[9,:]+[1,1]

grid[10,1]-=1.5
grid[11,1]-=0.5
grid[15,1]+=1
grid[16,1]+=1
grid[20,1]-=1

grid[24,1]+=0.5
grid[28,0]+=1.5
grid[29,:]=grid[29,:]+[-0.5,0.5]
grid[33,1]+=2
grid[41,1]+=1
grid[50,:]=grid[50,:]+[-1,0.5]

grid_delete=[21,22,27,32,36,37,40,43,44,45,57]
mask = np.ones(len(grid), dtype=bool)
mask[grid_delete] = False
grid_new= grid[mask]

additional_edges=np.array([[9,17]])

updated_edges=np.vstack((edges,additional_edges))

#updated_edges=edges

edges_to_delete =  np.array([[2,3],
                            [4,7],
                            [7,9],
                            [10,12],                                                    
                            [17,24],
                            [18,24],                           
                            [24,29],                       
                            [30,31],
                            [30,35],
                            [50,51],
                            [57,65],
                            [70,76],
                            [71,76],
                            [71,77]])

mask_new = np.array([not np.any(np.all(edge == edges_to_delete, axis=1)) for edge in updated_edges])


edges_new=updated_edges[mask_new]

# Creating a mapping of old indexes to new indexes
mapping = {old_idx: new_idx for new_idx, old_idx in enumerate([i for i in range(len(grid)) if i not in grid_delete])}

valid_edges = np.array([edge for edge in edges_new-1 if all(node in mapping for node in edge)])
# Adjusting edges based on new indexes

edges_new = np.vectorize(mapping.get)(valid_edges)+1

sp_digraph = SpatialDiGraph(genotypes, coord, grid_new, edges_new)

projection = ccrs.EquidistantConic(central_longitude=40)

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
        cbar_font_size=5,cbar_loc='lower center', cbar_ticklabelsize=8, 
        cbar_bbox_to_anchor=(0.0, -0.3, 1, 1),   
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


