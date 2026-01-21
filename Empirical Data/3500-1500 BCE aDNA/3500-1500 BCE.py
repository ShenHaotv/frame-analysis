import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from frame.spatial_digraph import SpatialDiGraph
from frame.visualization import Vis
from frame.cross_validation import run_cv
from frame.digraphstats import Digraphstats

"""To get the genotype file, please download the v62.0_1240k_public dataset from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FFIDCW, apply the filters we showed in the supplementary, delete the repeating samples (for this step, we suggest using 3500-1500 BCE_individual_list.csv), and do the mean value imputation """

"""For the coord, grid, edges file, please use 3500-1500 BCE.coord, 3500-1500 BCE_grid.csv and 3500-1500 BCE_edges.csv """

lamb_m_warmup = 1e3

lamb_m_grid = np.geomspace(1e-3,1e3,13)[::-1]

cv_errs,node_train_idxs=run_cv(sp_digraph,
                               lamb_m_grid=lamb_m_grid,
                               lamb_m_warmup=lamb_m_warmup,
                               n_folds=10,
                               factr=1e10,
                               random_state=100,
                               outer_verbose=True,
                               inner_verbose=False,)

if np.argmin(cv_errs)==0:
   lamb_m_grid_fine=np.geomspace(lamb_m_grid[0],lamb_m_grid[1],7)[::-1]

elif np.argmin(cv_errs)==12:
     lamb_m_grid_fine=np.geomspace(lamb_m_grid[11],lamb_m_grid[12], 7)[::-1]
     
else:
    lamb_m_grid_fine=np.geomspace(lamb_m_grid[np.argmin(cv_errs)-1],lamb_m_grid[np.argmin(cv_errs)+1], 7)[::-1]

cv_errs_fine,node_train_idxs_fine=run_cv(sp_digraph,
                                         lamb_m_grid=lamb_m_grid_fine,
                                         lamb_m_warmup=lamb_m_warmup,
                                         n_folds=10,
                                         factr=1e10,
                                         random_state=100,
                                         outer_verbose=True,
                                         inner_verbose=False,
                                         node_train_idxs=node_train_idxs)

lamb_m_opt=lamb_m_grid_fine[np.argmin(cv_errs_fine)]
lamb_m_opt=float("{:.3g}".format(lamb_m_opt))

sp_digraph.fit(lamb_m=lamb_m_warmup, factr=1e10)
logm = np.log(sp_digraph.m)
logc=np.log(sp_digraph.c)
trans_alpha=-np.log((1/sp_digraph.alpha)-1)

sp_digraph.fit(lamb_m=lamb_m_opt,
               factr=1e7,
               logm_init=logm,
               logc_init=logc,
               trans_alpha_init=trans_alpha)

projection = ccrs.Mercator()

fig, axs= plt.subplots(2, 4, figsize=(15, 6), dpi=300,
                        subplot_kw={'projection': projection})

v = Vis(axs[0,0], sp_digraph, projection=projection, edge_width=1,
        edge_alpha=1, edge_zorder=100, sample_pt_size=5,
        obs_node_size=0.5, sample_pt_color="black",
        cbar_font_size=8,cbar_ticklabelsize=8, 
        cbar_bbox_to_anchor=(0.4, -0.4, 1, 1),  
        compass_radius=0.3,
        compass_font_size=8,
        compass_bbox_to_anchor=(0.4, -0.5),
        mutation_scale=6)

v.digraph_wrapper(axs, node_scale=[5, 5, 10])
plt.subplots_adjust(hspace=0.1)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(np.log10(lamb_grid), cv, 'bo')  
plt.plot(np.log10(lamb_grid_fine), cv_fine, 'bo')  
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
