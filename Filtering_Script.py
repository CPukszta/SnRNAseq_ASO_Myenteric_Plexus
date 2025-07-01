import numpy as np
import numba
import pandas as pd
import anndata as ad

import iqplot
import scipy.integrate
import anndata
import scanpy as sc
import matplotlib.pyplot as plt # import matplotlib to visualize our qc metrics
from matplotlib.patches import Rectangle
import seaborn as sns
import well_plate
import plotly.io as plotly

from scipy.io import mmread
from matplotlib.pyplot import rc_context
import warnings

import bokeh.io
from bokeh.models import ColumnDataSource

bokeh.io.output_notebook()

#default filtering metrics:
n_Ercc = 400 #take cells with above this many ERCC counts
mt_pct = 10 #cut off for mitochondrial genes
n_UMI = 800 #cut off for number of UMIS, this number should be considered
n_UMI_high = 30000
Ercc_pct = 70

dir = "Filtering/All_Plate_Figures/"

def total_counts_plot(adata):
    str = "cell_counts"
    graph_data = adata.obs[str]
    filt = pd.DataFrame({str: graph_data, 'Status':['Filtered' if x == 0 else 'Retained' for x in adata.obs['Filtered_Status']]})
    
    umipc = sns.histplot(data = filt,x=str,hue="Status",bins=200,log_scale=True)
    umipc.axvline(x = n_UMI,    # Line on x = 2
               ymin = 0, # Bottom of the plot
               ymax = 1,
                color='red') # Top of the plot
    
    umipc.axvline(x = n_UMI_high,    # Line on x = 2
               ymin = 0, # Bottom of the plot
               ymax = 1,
                color='red') # Top of the plot

    # Create a Rectangle patch
    rect = Rectangle((n_UMI_high,0),max(graph_data) + 100000,400,linewidth=1,edgecolor='grey',facecolor='grey',alpha=0.2)
    size = 700
    rect_2 = Rectangle((n_UMI-size,0),size,400,linewidth=1,edgecolor='grey',facecolor='grey',alpha=0.2)

    #change plot labels
    umipc.set(xlabel='Number of UMI')
    
    # Add the patch to the Axes
    umipc.add_patch(rect)
    umipc.add_patch(rect_2)

    #remove the legend
    umipc.get_legend().remove()
    
    plt.savefig(dir+'UMI_Bargraph.png')
    plt.clf()

def pct_mt_plot(adata):
    str = "pct_counts_mt"
    graph_data = adata.obs[str]
    filt = pd.DataFrame({str: graph_data, 'Status':['Filtered' if x == 0 else 'Retained' for x in adata.obs['Filtered_Status']]})
    mtpc = sns.histplot(data = filt,x=str,hue="Status",bins=1000)
    mtpc.axvline(x=mt_pct,color='red')

    # Create a Rectangle patch
    rect = Rectangle((mt_pct,0),max(graph_data),400,linewidth=1,edgecolor='grey',facecolor='grey',alpha=0.2)

    #change plot labels
    mtpc.set(xlabel='Percent Mitochondrial Counts')
    
    # Add the patch to the Axes
    mtpc.add_patch(rect)

    #remove the legend
    mtpc.get_legend().remove()

    plt.savefig(dir+'Mito_Bargraph.png')
    plt.clf()
    
def total_ERCC_plot(adata):
    str = "total_counts_ERCC"
    graph_data = adata.obs[str]
    filt = pd.DataFrame({str: graph_data, 'Status':['Filtered' if x == 0 else 'Retained' for x in adata.obs['Filtered_Status']]})
    
    ERCCpc = sns.histplot(data = filt,x=str,hue="Status",bins=75)
    ERCCpc.axvline(x=n_Ercc,color='red')

    rect = Rectangle((n_Ercc-10000,0),10000,400,linewidth=1,edgecolor='grey',facecolor='grey',alpha=0.2)

    #change plot labels
    ERCCpc.set(xlabel='Total ERCC Counts')
    
    # Add the patch to the Axes
    ERCCpc.add_patch(rect)

    #remove the legend
    ERCCpc.get_legend().remove()

    
    plt.savefig(dir+'ERCC_Bargraph.png')
    plt.clf()


def pct_ERCC_plot(adata):
    str = "pct_counts_ERCC"
    graph_data = adata.obs[str]
    filt = pd.DataFrame({str: graph_data, 'Status':['Filtered' if x == 0 else 'Retained' for x in adata.obs['Filtered_Status']]})
    
    ERCCpct = sns.histplot(data = filt,x=str,hue="Status",bins=75)
    ERCCpct.axvline(x=Ercc_pct,color='red')

    rect = Rectangle((Ercc_pct,0),max(graph_data),400,linewidth=1,edgecolor='grey',facecolor='grey',alpha=0.2)

    #change plot labels
    ERCCpct.set(xlabel='Percent ERCC Counts')
    
    # Add the patch to the Axes
    ERCCpct.add_patch(rect)

    #remove the legend
    ERCCpct.get_legend().remove()
    
    plt.savefig(dir+'ERCC_pct_Bargraph.png')
    plt.clf()

def knee(adata,color='#1f77b4',box=True):
    knee = bokeh.plotting.figure(width = 362*2, height = 270*2,
                          title = "Knee Plot",
                          x_axis_label = "Cell Number", y_axis_label="Number of UMI",y_axis_type="log")
    x = np.linspace(1,np.shape(adata.obs["cell_counts"])[0],np.shape(adata.obs["cell_counts"])[0])
    
    UMI_sum_Sorted_filt = np.flip(np.sort(adata.obs["cell_counts"],axis=0))
    knee.scatter(x,UMI_sum_Sorted_filt,color=color,size = 3,marker='dot')
    
    knee.line([0,np.shape(adata.obs["cell_counts"])[0]],n_UMI,color='red')
    knee.line([0,np.shape(adata.obs["cell_counts"])[0]],n_UMI_high,color='red')

    if box == True:
        from bokeh.models import BoxAnnotation
        high_box = BoxAnnotation(bottom=n_UMI_high, fill_alpha=0.2, fill_color='grey')
        low_box = BoxAnnotation(top=n_UMI, fill_alpha=0.2, fill_color='grey')
        knee.add_layout(high_box)
        knee.add_layout(low_box)

    # bokeh.io.export_png(knee, filename=dir+"Knee_Plot.svg")
    knee.xgrid.visible = False
    knee.ygrid.visible = False

    knee.outline_line_width = 1
    # knee.outline_line_alpha = 0.3
    knee.outline_line_color = "black"
    
    bokeh.io.show(knee)


def knee_colored(adata,color='#1f77b4',box=True):
    knee = bokeh.plotting.figure(width = 362*2, height = 270*2,
                          title = "Knee Plot",
                          x_axis_label = "Cell Number", y_axis_label="Number of UMI",y_axis_type="log")
    x = np.linspace(1,np.shape(adata.obs["cell_counts"])[0],np.shape(adata.obs["cell_counts"])[0])

    #get the data
    df = adata.obs[["cell_counts","Filtered_Status"]].sort_values("cell_counts",ascending = False)
    df["Position"] = x

    #making dataframes for removed and retained cells
    df_retained = df[df["Filtered_Status"]]
    df_removed = df[~df["Filtered_Status"]]
    source_retained = bokeh.models.ColumnDataSource(df_retained)
    source_removed = bokeh.models.ColumnDataSource(df_removed)

    print(df_removed)

    #plot each of the types of cells
    # knee.scatter("Position","cell_counts",color='#1f77b4',size = 3,marker='dot',source=source_retained)
    knee.scatter("Position","cell_counts",color='red',size = 3,marker='dot',source=source_removed) ##ff7f0e
    
    knee.line([0,np.shape(adata.obs["cell_counts"])[0]],n_UMI,color='red')
    knee.line([0,np.shape(adata.obs["cell_counts"])[0]],n_UMI_high,color='red')

    if box == True:
        from bokeh.models import BoxAnnotation
        high_box = BoxAnnotation(bottom=n_UMI_high, fill_alpha=0.2, fill_color='grey')
        low_box = BoxAnnotation(top=n_UMI, fill_alpha=0.2, fill_color='grey')
        knee.add_layout(high_box)
        knee.add_layout(low_box)

    # bokeh.io.export_png(knee, filename=dir+"Knee_Plot.svg")
    knee.xgrid.visible = False
    knee.ygrid.visible = False

    knee.outline_line_width = 1
    knee.outline_line_color = "black"
    
    bokeh.io.show(knee)


def Flow_vs_counts(adata,color,n,flow_name):
    color_list = ["orange",color]
    plt = bokeh.plotting.figure(width = 700, height = 450,
                          title = "Flow Data vs Counts Plate {}".format(n),
                          x_axis_label = flow_name, y_axis_label="UMI Sum",y_axis_type="log",x_axis_type="log")
    plt.add_layout(bokeh.models.Legend(), 'right')
    data = ColumnDataSource(adata.obs) #input metadata into bokeh
    FILTERED = sorted(adata.obs["Filtered_Status"].unique())
    
    plt.circle(x=flow_name, y="total_counts",
               color=bokeh.transform.factor_cmap("Filtered_Status",color_list,FILTERED),source= data,legend_group ="Filtered_Status")    
    plt.line([adata.obs[flow_name].min(),adata.obs[flow_name].max()],n_UMI,color='red')
    plt.line([adata.obs[flow_name].min(),adata.obs[flow_name].max()],n_UMI_high,color='red')

    bokeh.io.show(plt)

def Save_adata(adata,n):
    adata = adata[combined_mask]

    #Save the object after filtering
    adata.write('Filtering/plate{}_filt.h5ad'.format(n), compression="gzip")


def scanpy_filtering(adata,n_Ercc = n_Ercc,mt_pct = mt_pct,n_UMI = n_UMI,n_UMI_high = n_UMI_high,Ercc_pct = Ercc_pct):
    'takes in the anndata object and filters it based on the metrics given'
      
    #preprocessing
    adata.var['ERCC'] = adata.var_names.str.startswith('ERCC-')
    adata.var['mt'] = adata.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    adata.var['ribosomal'] = adata.var_names.str.startswith('rp')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars = ['ERCC','mt','ribosomal'],inplace=True) #run QC and save it to the andata object
    
    #producing masks for each filter
    ERCC_mask = adata.obs['total_counts_ERCC'] > n_Ercc #remove too low ERCCs
    ERCC_mask_PCT = adata.obs['pct_counts_ERCC'] < Ercc_pct #remove too high of a percentage of mt genes
    MT_mask = adata.obs['pct_counts_mt'] < mt_pct #remove too high % of mitochondrial genes
    UMI_mask,_ = sc.pp.filter_cells(adata, min_counts = n_UMI,inplace=False)
    UMI_mask_high,_ = sc.pp.filter_cells(adata, max_counts = n_UMI_high,inplace=False) #remove too high UMI
    empty_well = np.invert(pd.isna(adata.obs['Genotype'])) #remove empty wells
    combined_mask = np.logical_and.reduce((ERCC_mask, ERCC_mask_PCT, MT_mask,UMI_mask,UMI_mask_high,empty_well))

    adata.obs["Filtered_Status"] = combined_mask
    return adata


    