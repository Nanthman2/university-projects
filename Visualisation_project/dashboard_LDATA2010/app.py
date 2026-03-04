###############################################################################
# Dashboard imports
###############################################################################

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import shinyswatch

# path
from pathlib import Path
import sys

project_root = Path(__file__).parent  
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print(f"Project root: {project_root}")

# data wrangling
import pandas as pd
import numpy as np
import json

# visualization
import plotly.express as px
from src.color_plot_config import OKABE_ITO_COLORS

from src.univariate_plots import plot_univariate
from src.bivariate_plots import plot_bivariate
from src.utils import detect_and_set_types, fill_categorical_missing_with_UNK, format_label
from src.dimensionality_reduction import *
from src.gene_expression import create_volcano_plot
from src.cluster_and_plots import *
from src.QC import metric_card, get_quality_metrics, plot_missingness
from src.dbcv_manual import calculate_dbcv_manual

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from joblib import parallel_backend
import time 
from src.dimensionality_reduction import (prepare_feature_matrix, estimate_all_dimensions, compute_umap, compute_pca, compute_neighborhood_preservation,compute_auc_metric)
from sklearn.manifold import trustworthiness
import traceback

# Load data
df = pd.read_csv(project_root / "data" / "METABRIC_RNA_Mutation.csv")

# Keep a copy for QC
df_unprocessed = df.copy()

###############################################################################
# VARIABLE DEFINITIONS
###############################################################################

df = df.drop(columns='cancer_type') # only 2 categories, with 1 containing only a single patient

FULL_VARIABLE_LIST = df.columns.tolist()

EXPERIMENTAL_VARIABLES = ['patient_id','cohort']
DEMOGRAPHIC_VARIABLES = ['age_at_diagnosis','inferred_menopausal_state']
INTERVENTION_VARIABLES = ['type_of_breast_surgery','chemotherapy','hormone_therapy','radio_therapy']
OUTCOME_VARIABLES = ['overall_survival_months', 'overall_survival', 'death_from_cancer', 'mutation_count']
MUTATION_VARIABLES = [col for col in FULL_VARIABLE_LIST if col.endswith('_mut')]
TUMOR_CLINICAL_FEATURES = ['cancer_type_detailed','tumor_other_histologic_subtype','cellularity','neoplasm_histologic_grade','tumor_size','tumor_stage','primary_tumor_laterality','lymph_nodes_examined_positive', 'nottingham_prognostic_index','oncotree_code']
RECEPTOR_STATUS = ['er_status','er_status_measured_by_ihc','pr_status','her2_status','her2_status_measured_by_snp6']
MOLECULAR_CLASSIFICATION = ['pam50_+_claudin-low_subtype','3-gene_classifier_subtype','integrative_cluster']
GENE_EXPRESSION_VARIABLES = [
    col for col in FULL_VARIABLE_LIST 
    if col not in (
        EXPERIMENTAL_VARIABLES + DEMOGRAPHIC_VARIABLES + INTERVENTION_VARIABLES + 
        TUMOR_CLINICAL_FEATURES + RECEPTOR_STATUS + MOLECULAR_CLASSIFICATION + 
        OUTCOME_VARIABLES + MUTATION_VARIABLES
    ) 
]

###############################################################################
# PREPROCESSING 
###############################################################################

df = df.drop(columns=list(MUTATION_VARIABLES))
detect_and_set_types(df)
fill_categorical_missing_with_UNK(df)

###############################################################################
# VARIABLE CHOICES GENERATION
###############################################################################

def get_variable_choices():
    choices = {}
    if DEMOGRAPHIC_VARIABLES:
        choices["Demographic"] = {var: format_label(var) for var in DEMOGRAPHIC_VARIABLES}
    if INTERVENTION_VARIABLES:
        choices["Intervention"] = {var: format_label(var) for var in INTERVENTION_VARIABLES}
    if OUTCOME_VARIABLES:
        choices["Outcome"] = {var: format_label(var) for var in OUTCOME_VARIABLES}
    if TUMOR_CLINICAL_FEATURES:
        choices["Tumor Clinical"] = {var: format_label(var) for var in TUMOR_CLINICAL_FEATURES}
    if RECEPTOR_STATUS:
        choices["Receptor Status"] = {var: format_label(var) for var in RECEPTOR_STATUS}
    if MOLECULAR_CLASSIFICATION:
        choices["Molecular Classification"] = {var: format_label(var) for var in MOLECULAR_CLASSIFICATION}
    if GENE_EXPRESSION_VARIABLES:
        choices["Gene Expression"] = {var: var.upper() for var in GENE_EXPRESSION_VARIABLES}
    return choices

def get_groupby_choices():
    choices = {"none": "None"}
    for var in MOLECULAR_CLASSIFICATION:
        choices[var] = format_label(var)
    for var in RECEPTOR_STATUS:
        choices[var] = format_label(var)
    for var in TUMOR_CLINICAL_FEATURES:
        choices[var] = format_label(var)
    for var in OUTCOME_VARIABLES:
        choices[var] = format_label(var)
    return choices

def get_clinical_choices():
    choices = get_variable_choices().copy()
    if "Gene Expression" in choices:
        del choices["Gene Expression"]
    return choices

##############################################################################
# IMPORT DATA DICTIONARY
###############################################################################

with open(project_root / "data" / "data_dictionary.json") as f:
    VARIABLE_DESCRIPTIONS = json.load(f)

##############################################################################
# SHINY APPLICATION - UI
##############################################################################

app_ui = ui.page_navbar(
    # ========================================================================
    # OVERVIEW PANE
    # ========================================================================
    ui.nav_panel(
        "Overview",
        ui.navset_card_pill(
            ui.nav_panel(
                "Stakes of the Dashboard",
                ui.markdown("""
                    This dashboard provides an **interactive exploration tool** for the **METABRIC dataset**, which contains molecular and clinical profiles for approximately **1,900 breast cancer patients**.<br><br>
                            
                    Traditional breast cancer classification relies primarily on:
                    - **Tumor morphology** (histological examination)
                    - **Receptor status** (ER, PR, HER2)

                    However, these criteria **do not fully capture the molecular heterogeneity** of breast tumors. Understanding this diversity is essential to improve both **prognosis** and **treatment selection**.<br><br>

                    Because genomic data is highly dimensional (our body alone contains ~20,000 genes), techniques such as **dimensionality reduction** and **clustering** are useful for identifying potential new molecular subgroups.<br><br>
                      
                    The dashboard is designed to be explored sequentially, from left to right across the navigation tabs. That said, you can dive into any section directly based on what you want to discover:

                    * **Exploratory Data Analysis** : How do clinical features relate to molecular profiles? Are some genes more expressed than others ?
                    * **Dimensionality Reduction** : Can 331 genes be meaningfully visualized in 2D/3D space? Do these sub-spaces allow to recover known groups ?
                    * **Clustering Analysis** : Do patients naturally group into distinct molecular subtypes?<br><br>
                        
                    **Two complementary approaches:**<br><br>
  
                    It offers both a **Learning Mode 🎓** (guided exploration with educational context) and **Expert Mode 🔬** (full parameter control for advanced users). You can **toggle between modes at any time** using the buttons in the top-right corner, so feel free to switch perspectives as you explore! <br><br>
                        
                    **Ready to dive in?** Start with the **Overview** tab to familiarize yourself with the dataset, then explore the various analytical tools at your own pace ! Enjoy the journey through the data!        
                """)
            ),

            ui.nav_panel(
                "About METABRIC",
                ui.markdown("""
                    The *Molecular Taxonomy of Breast Cancer International Consortium* (METABRIC) dataset was collected from tumors sampled between **1977 and 2005** across five medical centers in the UK and Canada. It was originally published by [Curtis et al., 2012](https://www.nature.com/articles/nature10983) and is publicly available through [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric).<br><br>

                    To facilitate intuitive exploration, variables have been organized into **biologically and clinically meaningful categories**.
                """),
                ui.br(),
                ui.card(
                    ui.input_selectize(
                        "variable_category",
                        "Select a category:",
                        choices={
                            "demographic": f"Demographic ({len(DEMOGRAPHIC_VARIABLES)})",
                            "intervention": f"Intervention ({len(INTERVENTION_VARIABLES)})",
                            "tumor_clinical": f"Tumor Clinical Features ({len(TUMOR_CLINICAL_FEATURES)})",
                            "receptor": f"Receptor Status ({len(RECEPTOR_STATUS)})",
                            "molecular": f"Molecular Classification ({len(MOLECULAR_CLASSIFICATION)})",
                            "outcome": f"Outcome ({len(OUTCOME_VARIABLES)})",
                            "genes": f"Gene Expression ({len(GENE_EXPRESSION_VARIABLES)})"
                        },
                        selected="demographic"
                    ),
                    ui.output_table("variable_description_table")  
                ),
            ),
            
            ui.nav_panel(
                "Data Quality Check",
                ui.layout_column_wrap(

                    ui.card(
                        ui.card_header("Preprocessing Steps"),
                        ui.markdown("""
                            The METABRIC dataset is already extensively preprocessed and quality-controlled. However, some additional steps were performed to enhance downstream analysis:
                
                            **Variables Exclusion**
                            * Mutation columns (`*_mut`) : sparse categorical variables
                            * `cancer_type` : single "Breast Sarcoma" instance, remainder classified as "Breast Cancer"
                            
                            **Missing Values Removal**
                            * Preserved to account for *informative missingness*
                            * Mapped to `UNK` category for categorical variables
                                    
                            The visuals on the right display the characteristics of the **RAW** data.
                        """),
                    ),
                    
                    ui.div(
                        ui.layout_column_wrap(
                            metric_card("Missing Values", "qc_missing"),
                            metric_card("Duplicates", "qc_duplicates"),
                            width=1/2
                        ),
                        ui.br(),
                        ui.card(
                            ui.card_header("Missingness Pattern"),
                            output_widget("qc_missingness_plot"),
                        ),
                    ),

                    width=1/2
                )
            )
        ),
    ),  
    # ========================================================================
    # EXPLORATORY ANALYSIS
    # ========================================================================
    ui.nav_menu(
        "Exploratory Analysis",
        ui.nav_panel(
            "Univariate",
            ui.layout_column_wrap(
                ui.card(
                    ui.layout_column_wrap(
                        ui.input_selectize("uni_var_1", "Variable", choices=get_variable_choices(), selected="age_at_diagnosis"),
                        ui.input_selectize("uni_group_1", "Group By (optional)", choices=get_groupby_choices(), selected="none"),
                        width=1/2
                    ),
                    output_widget("univariate_plot_1")
                ),
                ui.card(
                    ui.layout_column_wrap(
                        ui.input_selectize("uni_var_2", "Variable", choices=get_variable_choices(), selected="type_of_breast_surgery"),
                        ui.input_selectize("uni_group_2", "Group By (optional)", choices=get_groupby_choices(), selected="none"),
                        width=1/2
                    ),
                    output_widget("univariate_plot_2")
                ),
                width=1/2, heights_equal="row"
            )
        ),
        
        ui.nav_panel(
            "Bivariate",
            ui.layout_column_wrap(
                ui.card(
                    ui.layout_column_wrap(
                        ui.input_selectize("bi_x_1", "X", choices=get_variable_choices(), selected="age_at_diagnosis"),
                        ui.input_selectize("bi_y_1", "Y", choices=get_variable_choices(), selected="tumor_size"),
                        ui.input_selectize("bi_color_1", "Color", choices=get_groupby_choices(), selected="none"),
                        width=1/3
                    ),
                    output_widget("bivariate_plot_1")
                ),
                ui.card(
                    ui.layout_column_wrap(
                        ui.input_selectize("bi_x_2", "X", choices=get_variable_choices(), selected="pam50_+_claudin-low_subtype"),
                        ui.input_selectize("bi_y_2", "Y", choices=get_variable_choices(), selected="neoplasm_histologic_grade"),
                        width=1/2
                    ),
                    output_widget("bivariate_plot_2")
                ),
                width=1/2, heights_equal="row"
            )
        ),

        ui.nav_panel("Gene Expression", ui.output_ui("gene_expression_layout")),
    ),

    # ========================================================================
    # DIMENSIONALITY REDUCTION
    # ========================================================================
    ui.nav_panel(
        "Dimensionality Reduction",
        ui.layout_sidebar(
            ui.sidebar(
                # INVISIBLE BRIDGE (to acces user mode)
                ui.div(
                    ui.input_radio_buttons("mode_bridge", "Bridge", choices=["learner", "expert"], selected="learner"),
                    style="display: none;"
                ),
                
                ui.tags.h5("Global Parameters"),
                ui.tooltip(
                    ui.input_radio_buttons("dr_n_components", "Projection Space:", choices={"2": "2D", "3": "3D"}, selected="2", inline=True),
                    "Dimensionality of the embedding space for visualization. 2D is faster, 3D preserves more structure.",
                    placement="right"
                ),

                ui.tooltip(
                    ui.input_slider("dr_sample_fraction", "Sample Size (%):", min=10, max=100, value=50, step=10, post="%"),
                    "Percentage of patients to include in the analysis. Lower values speed up computation.",
                    placement="right"
                ),

                ui.tooltip(
                    ui.input_checkbox("include_pca", "PCA benchmark", value=True),
                    "Include PCA projection alongside UMAP for comparison. Adds intrinsic dimensionality analysis.",
                    placement="right"
                ),

                ui.hr(),
                ui.tags.h5("UMAP Parameters"),

                ui.tooltip(
                    ui.input_slider("umap_n_neighbors", "N Neighbors:", min=5, max=100, value=15, step=5),
                    "Controls balance between local and global structure. Lower values (5-15) emphasize local patterns, higher values (50+) preserve global structure.",
                    placement="right"
                ),

                ui.tooltip(
                    ui.input_slider("umap_min_dist", "Min Distance:", min=0.0, max=0.99, value=0.1, step=0.05),
                    "Minimum distance between points in the embedding. Lower values create tighter clusters, higher values spread points out more evenly.",
                    placement="right"
                ),

                ui.panel_conditional(
                    "input.mode_bridge === 'expert'",
                    ui.TagList(
                        ui.tags.h5("🔬 Expert Parameters"),
                        
                        ui.tooltip(
                            ui.input_select("umap_metric", "Similarity Metric:", choices=["euclidean", "cosine", "manhattan", "chebyshev"], selected="euclidean"),
                            "Distance metric for computing similarity in high-dimensional space.",
                            placement="right"
                        ),
                        
                        ui.tooltip(
                            ui.input_select("auc_base_metric", "AUC Evaluation:", choices=['Trustworthiness', 'Continuity', 'R_NX'], selected="R_NX"),
                            "Metric for computing quality Area Under the Curve for a chosen metric. The score is stored for each run.",
                            placement="right"
                        ),
                        
                        ui.tooltip(
                            ui.input_numeric("umap_random_state", "Random State:", value=2003, min=0, max=9999),
                            "Seed for reproducibility.",
                            placement="right"
                        )
                    ),
                ),
                ui.input_action_button("run_dr", "Run", class_="btn-primary btn-lg", width="90%"),
                width=200
            ),
            ui.output_ui("dr_tabs_dynamic")
        )
    ),

    # ========================================================================
    # CLUSTERING
    # ========================================================================
    ui.nav_menu(
            "Clustering",
        
            # K-MEANS
            ui.nav_panel(
                "K-Means",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.div(ui.input_radio_buttons("mode_bridge_kmeans", "Mode", ["learner", "expert"], selected="learner"), style="display: none;"),
                        ui.tags.h5("1. Data & Preprocessing"),
                        ui.input_action_button("import_dr_kmeans", "⬇️ Import settings from previous DR", class_="btn-secondary btn-sm mb-3"),
                        ui.tooltip(
                        ui.input_slider("clust_sample_fraction", "Sample Size (%):", min=10, max=100, value=50, step=10, post="%"),
                        "Percentage of patients to include. Lower values speed up computation !",
                        placement="right"
                        ),
                        ui.tooltip(
                        ui.input_select("clust_prep_method", "Gene Reduction Strategy:", choices={"none": "None (Raw Data)", "pca": "PCA", "umap": "UMAP"}, selected="pca"),
                        "Technique to reduce the 489 genes into fewer dimensions to avoid the Curse of Dimensionality.",
                        placement="right"
                        ),
                        ui.panel_conditional("input.clust_prep_method === 'pca'", ui.input_numeric("clust_prep_pca_n", "Keep N Dimensions:", value=50, min=2, max=331)),
                        ui.panel_conditional("input.clust_prep_method === 'umap'",
                            ui.input_numeric("clust_prep_umap_n", "Keep N Dimensions:", value=10, min=2, max=50),
                            ui.input_slider("clust_prep_umap_neighbors", "N Neighbors:", min=5, max=100, value=30),
                            ui.input_slider("clust_prep_umap_mindist", "Min Dist:", min=0.0, max=0.99, value=0.0, step=0.05)
                        ),
                        ui.hr(),
                        ui.tags.h5("2. K-Means Parameters"),
                        ui.tooltip(
                        ui.input_slider("clust_k", "Number of Clusters (k):", min=2, max=20, value=3, step=1),
                        "The target number of groups you want to find. Use the CH/Gap plots to find the best value.",
                        placement="right"
                        ),
                        ui.panel_conditional("input.mode_bridge === 'expert'",
                        ui.tooltip(
                        ui.input_numeric("clust_n_init", "N Initializations (n_init)(🔬 Expert Parameters):", value=10, min=1, step=1),
                        "Number of times the algorithm will run with different starting points. It keeps the best result (lowest error). Useful to avoid bad random starts.",
                        placement="right"
                        )
                        ),
                        ui.hr(),
                        ui.panel_conditional(
                            "input.clust_prep_method === 'none' || " + \
                            "(input.clust_prep_method === 'pca' && input.clust_prep_pca_n > 3) || " + \
                            "(input.clust_prep_method === 'umap' && input.clust_prep_umap_n > 3)",
                        
                            ui.tags.h5("3. Visualization"),
                            ui.input_radio_buttons("clust_proj_dims", "Plot Dimensions:", choices={"2": "2D", "3": "3D"}, selected="2", inline=True),
                            ui.input_select("clust_viz_method", "Projection Method:", choices={"pca": "PCA", "umap": "UMAP"}, selected="pca"),
                            ui.panel_conditional("input.clust_viz_method === 'umap'",
                                ui.input_slider("clust_viz_umap_neighbors", "N Neighbors:", min=5, max=50, value=15),
                                ui.input_slider("clust_viz_umap_min_dist", "Min Dist:", min=0.0, max=0.99, value=0.1, step=0.05)
                            )
                        ),
                    
                        ui.br(),
                        ui.input_action_button("run_clustering", "Run", class_="btn-primary btn-lg", width="100%"),width=300
                    ),
                    ui.output_ui("kmeans_tabs_dynamic")
                )
            ),
        
            # HIERARCHICAL
            ui.nav_panel(
                "Hierarchical",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.tags.h5("1. Data & Preprocessing"),
                        ui.input_action_button("import_dr_hc", "⬇️ Import settings from previous DR", class_="btn-secondary btn-sm mb-3"),
                        ui.tooltip(
                        ui.input_slider("hc_sample_fraction", "Sample Size (%):", min=5, max=100, value=50, step=5, post="%"),
                        "Percentage of patients to include. Lower values speed up computation !",
                        placement="right"
                        ),                  
                        ui.input_select("hc_prep_method", "Preprocessing Strategy:", choices={"none": "None (Raw)", "pca": "PCA", "umap": "UMAP"}, selected="pca"),
                        ui.panel_conditional("input.hc_prep_method === 'pca'", ui.input_numeric("hc_prep_pca_n", "PCA Dims:", value=30)),
                        ui.panel_conditional("input.hc_prep_method === 'umap'",
                            ui.input_numeric("hc_prep_umap_n", "UMAP Dims:", value=5),
                            ui.input_slider("hc_prep_umap_neighbors", "N Neighbors:", min=5, max=100, value=25),
                            ui.input_slider("hc_prep_umap_mindist", "Min Dist:", min=0.0, max=0.99, value=0.1, step=0.05)
                        ),
                        ui.hr(),
                        ui.tags.h5("2. Clustering Params"),
                        ui.tooltip(
                        ui.input_select("hc_linkage", "Linkage:", choices={"centroid": "Centroid", "complete": "Complete (Max linkage)"}, selected="centroid"),
                        "Method to calculate distance between two clusters. 'Complete' tends to find compact, spherical clusters.",
                        placement="right"
                        ),                    
                        ui.tooltip(
                        ui.input_slider("hc_k", "Number of Clusters (k):", min=2, max=20, value=4),
                        "Determines where to 'cut' the dendrogram tree to form discrete groups.",
                        placement="right"
                        ),

                        ui.hr(),
                        ui.panel_conditional(
                            "input.hc_prep_method === 'none' || " + \
                            "(input.hc_prep_method === 'pca' && input.hc_prep_pca_n > 3) || " + \
                            "(input.hc_prep_method === 'umap' && input.hc_prep_umap_n > 3)",
                        
                            ui.tags.h5("3. Visualization"),
                            ui.input_radio_buttons("hc_proj_dims", "Plot Dimensions:", choices={"2": "2D", "3": "3D"}, selected="2", inline=True),
                            ui.input_select("hc_viz_method", "Projection Method:", choices={"pca": "PCA", "umap": "UMAP"}, selected="pca"),
                            ui.panel_conditional("input.hc_viz_method === 'umap'",
                                ui.input_slider("hc_viz_umap_neighbors", "N Neighbors:", min=5, max=50, value=15),
                                ui.input_slider("hc_viz_umap_min_dist", "Min Dist:", min=0.0, max=0.99, value=0.1, step=0.05)
                            )
                        ),
                        ui.br(),
                        ui.input_action_button("run_hc", "Run", class_="btn-primary btn-lg", width="100%"),
                        width=320
                    ),
                    ui.output_ui("hc_tabs_dynamic")

                )
            ),
        
            # DBSCAN
            ui.nav_panel(
                "DBSCAN",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.tags.h5("1. Data & Preprocessing"),
                        ui.input_action_button("import_dr_db", "⬇️ Import settings from previous DR", class_ = "btn-secondary btn-sm mb-3"),
                        ui.tooltip(
                        ui.input_slider("db_sample_fraction", "Sample Size (%):", min=10, max=100, value=50, step=10, post="%"),
                        "Percentage of patients to include. Lower values speed up computation !",
                        placement="right"
                        ),                  
                        ui.input_select("db_prep_method", "Gene Reduction Strategy:", choices={"none": "None", "pca": "PCA", "umap": "UMAP"}, selected="pca"),
                        ui.panel_conditional("input.db_prep_method === 'pca'", ui.input_numeric("db_prep_pca_n", "Keep N Dimensions:", value=50, min=2, max=331)),
                        ui.panel_conditional("input.db_prep_method === 'umap'",
                            ui.input_numeric("db_prep_umap_n", "Keep N Dimensions:", value=10, min=2, max=50),
                            ui.input_slider("db_prep_umap_neighbors", "N Neighbors:", min=5, max=100, value=30),
                            ui.input_slider("db_prep_umap_mindist", "Min Dist:", min=0.0, max=0.99, value=0.0, step=0.05)
                        ),
                        ui.hr(),
                        ui.tags.h5("2. DBSCAN Parameters"),
                        ui.tooltip(
                        ui.input_slider("db_eps", "Epsilon (Radius):", min=1.0, max=60.0, value=25.0, step=0.5),
                        "The maximum distance between two samples for one to be considered as in the neighborhood of the other. Use the 'Knee' plot to tune Epsilon.",
                        placement="right"
                        ),
                
                        ui.tooltip(
                        ui.input_slider("db_min_samples", "Min Samples (Density):", min=2, max=20, value=5, step=1),
                        "The number of samples in a neighborhood for a point to be considered as a core point (dense region).",
                        placement="right"
                        ),
                        ui.hr(),
                        ui.panel_conditional(
                            "input.db_prep_method === 'none' || " + \
                            "(input.db_prep_method === 'pca' && input.db_prep_pca_n > 3) || " + \
                            "(input.db_prep_method === 'umap' && input.db_prep_umap_n > 3)",
                        
                            ui.tags.h5("3. Visualization"),
                            ui.input_radio_buttons("db_proj_dims", "Plot Dimensions:", choices={"2": "2D", "3": "3D"}, selected="2", inline=True),
                            ui.input_select("db_viz_method", "Projection Method:", choices={"pca": "PCA", "umap": "UMAP"}, selected="pca"),
                            ui.panel_conditional("input.db_viz_method === 'umap'",
                                ui.input_slider("db_viz_umap_neighbors", "N Neighbors:", min=5, max=50, value=15),
                                ui.input_slider("db_viz_umap_min_dist", "Min Dist:", min=0.0, max=0.99, value=0.1, step=0.05)
                            )
                        ),
                        ui.br(),
                        ui.input_action_button("run_dbscan", "Run", class_="btn-primary btn-lg", width="100%"),
                        width=300
                    ),          
                    ui.output_ui("db_tabs_dynamic")

                )
            ),
        
            # COMPARISON
            ui.nav_panel(
                "Comparison",
                ui.card(
                    ui.card_header("Clustering Analysis History"),
                    ui.output_table("cluster_history_table"),
                    full_screen=True,
                    style="height: 300px; overflow-y: auto;"
                ),
                ui.br(),
                ui.card(
                    ui.output_ui("comparison_plots_layout"),
                    full_screen=True,
                    style="min-height: 500px;"
                )
            )
        ),
    # ========================================================================
    # TOP RIGHT BUTTONS
    # ========================================================================
    ui.nav_spacer(),
    ui.nav_control(ui.input_action_button("toggle_learning", "Learning", class_="btn-sm btn-outline-primary")),
    ui.nav_control(ui.input_action_button("toggle_expert", "Expert", class_="btn-sm btn-outline-primary")),

    title="METABRIC Dashboard",
    fillable=True,
    theme=shinyswatch.theme.zephyr,
)

##############################################################################
# SERVER
##############################################################################

def server(input, output, session):
    
    # ========================================================================
    # MODE MANAGEMENT (POPUP + BRIDGE)
    # ========================================================================
    first_visit = reactive.Value(True)
    active_mode = reactive.Value(None)
    recent_plots = reactive.Value([]) 
    @reactive.Effect
    def _():
        if first_visit.get():
            m = ui.modal(
                ui.div( # emojis taken from : https://gist.github.com/rxaviers/7360908
                    ui.tags.h5("How would you like to explore the data"),
                    ui.br(),
                    ui.markdown("""
                            * 🎓 **Learning Mode** : a guided workflow with educational context for users new to breast cancer genomics or data science.

                            * 🔬 **Expert Mode** : full parameter control with minimal guidance for experienced researchers and data scientists.
                            """)
                ),
                title="Welcome to the METABRIC Dashboard ! 👋",
                footer=ui.TagList(
                    ui.input_action_button("start_learning", "Start as Learner", class_="btn-success"),
                    ui.input_action_button("start_expert", "Start as Expert", class_="btn-primary")
                ),
                easy_close=False, size="l"
            )
            ui.modal_show(m)
            first_visit.set(False)

    @reactive.Effect
    @reactive.event(input.start_learning)
    def _():
        toggle_mode("learning")
        ui.modal_remove()

    @reactive.Effect
    @reactive.event(input.start_expert)
    def _():
        toggle_mode("expert")
        ui.modal_remove()

    def toggle_mode(mode):
        active_mode.set(mode)
        ui.update_action_button("toggle_learning", label="🎓 Learning" if mode == "learning" else "Learning")
        ui.update_action_button("toggle_expert", label="🔬 Expert" if mode == "expert" else "Expert")
        ui.update_radio_buttons("mode_bridge", selected="expert" if mode == "expert" else "learner")

    @reactive.effect
    @reactive.event(input.toggle_learning)
    def _(): toggle_mode("learning")

    @reactive.effect
    @reactive.event(input.toggle_expert)
    def _(): toggle_mode("expert")

    # ========================================================================
    # OVERVIEW - VARIABLE EXPLORER
    # ========================================================================
    @render.table(index=False)
    def variable_description_table():
        chosen_category = input.variable_category()
        category_map = {
            "demographic": DEMOGRAPHIC_VARIABLES, "intervention": INTERVENTION_VARIABLES,
            "tumor_clinical": TUMOR_CLINICAL_FEATURES, "receptor": RECEPTOR_STATUS,
            "molecular": MOLECULAR_CLASSIFICATION, "outcome": OUTCOME_VARIABLES
        }
        vars_list = category_map.get(chosen_category, [])
        table_data = []
        for var in vars_list:
            table_data.append({'Variable': format_label(var), 'Description': VARIABLE_DESCRIPTIONS.get(var, 'No description available')})
        
        if chosen_category == "genes":
            table = pd.DataFrame({
                'Variable': ['Gene Expression Variables'],
                'Description': [f"{len(GENE_EXPRESSION_VARIABLES)} genes with mRNA expression z-scores. Examples: " + 
                        ", ".join([g.upper() for g in GENE_EXPRESSION_VARIABLES[:10]]) + ", ..."]
            })
        else:
            table = pd.DataFrame(table_data) 
        
        return table.style.set_table_attributes('class="dataframe shiny-table table w-auto"')
    
    # ========================================================================
    # OVERVIEW - QC
    # ========================================================================
    @render.text
    def qc_duplicates():
        metrics = get_quality_metrics(df_unprocessed)
        return f"{metrics['n_duplicates']:,}"

    @render.text
    def qc_missing():
        metrics = get_quality_metrics(df_unprocessed)
        return f"{metrics['pct_missing']:.2f}%"

    @render_widget
    def qc_missingness_plot():
        return plot_missingness(df_unprocessed, drop_no_missing_cols=True)
    
    # ========================================================================
    # EDA - UNIVARIATE
    # ========================================================================
    @render_widget
    def univariate_plot_1():
        var = input.uni_var_1()
        group = input.uni_group_1()
        return plot_univariate(df, variable=var, group_by=group if group != 'none' else None)
    
    @render_widget
    def univariate_plot_2():
        var = input.uni_var_2()
        group = input.uni_group_2()
        return plot_univariate(df, var, group_by=group if group != 'none' else None)
    
  


    # ========================================================================
    # EDA - BIVARIATE
    # ========================================================================
    @render_widget
    def bivariate_plot_1():
        x_var = input.bi_x_1()
        y_var = input.bi_y_1()
        color = input.bi_color_1()
        return plot_bivariate(df, x_var=x_var, y_var=y_var, color_by=color if color != 'none' else None)
    
    @render_widget
    def bivariate_plot_2():
        x_var = input.bi_x_2()
        y_var = input.bi_y_2()
        return plot_bivariate(df, x_var=x_var, y_var=y_var)
    
    # ========================================================================
    # EDA - VOLCANO PLOT
    # ========================================================================
    volcano_data = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.compute_volcano)
    def compute_volcano():
        with ui.Progress() as p:
            try:
                p.set(message="[1/1] Computing fold changes and p-values...")
                fig = create_volcano_plot(
                    df=df, gene_vars=GENE_EXPRESSION_VARIABLES,
                    group_var=input.volcano_var(),
                    fc_thresh=input.fc_threshold(),
                    p_thresh=input.p_threshold()
                )
                volcano_data.set(fig)
                ui.notification_show("Done !", type="success", duration=10)
            except Exception as e:
                volcano_data.set(None)
                ui.notification_show(f"Error: {str(e)}", type="error", duration=5)

    @render.ui
    def volcano_plot_content():
        fig = volcano_data.get()
        if fig is None:
            return ui.div(
                ui.markdown("Configure parameters and click **Compute** to generate the volcano plot"),
                style="text-align: center; min-height: 400px; display: flex; align-items: center; justify-content: center;"
            )
        return ui.div(output_widget("volcano_widget_plot"), style="min-height: 400px;")

    @render_widget
    def volcano_widget_plot():
        return volcano_data.get()

    @render.ui
    def volcano_explanation_ui():
        if active_mode.get() != "learning":
            return None
        return ui.markdown("""
                    A volcano plot visualizes **differential gene expression analysis** between two groups A and B. 

                    *What happens under the hood?*

                    For each of the 331 genes in the dataset:

                    1. **Split patients** into two groups A and B
                    2. **Calculate mean expression** in both groups
                    3. **Compute fold-change:** Ratio mean(A) / mean(B) 
                    4. **Apply log2 transformation** to the fold-change (see why on this [Wikipedia](https://en.wikipedia.org/wiki/Fold_change) page)
                    5. **Perform statistical test:** H_0: no difference in mean expression between groups

                    *How to read the plot?*

                    Each point on the scatter plot represents a gene.

                    The x-coordinate represents the **log2(fold-change)**. It translates biological effect size.
                    - **Positive values** suggest that the gene is **more expressed in group A**
                    - **Negative values** suggest that the gene is **more expressed in group B**
                    - The default FC Threshold = 1 means a 2*difference (2^1 = 2)

                    The y-coordinate represents the **-log10(p-value)**. It translates statistical confidence.
                    - **Higher values** mean stronger evidence the difference is real, not random
                    - p = 0.05 corresponds to -log10(p) = 1.3
                    - p = 0.001 corresponds to -log10(p) = 3

                    *Key regions of the plot:*

                    - **Top-right:** Genes UP-regulated in A (higher expression + statistically significant)
                    - **Top-left:** Genes DOWN-regulated in A (lower expression + statistically significant)
                    - **Bottom:** Non-significant genes (small effect OR high uncertainty)

                    *How to choose my thresholds?*

                    - **FC Threshold:** Controls minimum log₂ fold-change to consider (default 1.0 = 2*change)
                    - **P-value Threshold:** Controls maximum p-value to accept (default 0.05 = 5% false positive rate)

                    **A gene is highlighted if it passes BOTH thresholds** (large effect AND statistically significant).
                    """)

    @render.ui
    def gene_expression_layout():
        is_learning = active_mode.get() == "learning"
        
        if is_learning:
            return ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Volcano Plot"),
                    ui.layout_column_wrap(
                        ui.input_select("volcano_var", "Variable", choices=['er_status','type_of_breast_surgery','chemotherapy','hormone_therapy','radio_therapy','primary_tumor_laterality','er_status_measured_by_ihc','pr_status','her2_status']),
                        ui.input_numeric("fc_threshold", "FC Threshold", value=1.0, min=0.1, max=3.0, step=0.1),
                        ui.input_numeric("p_threshold", "P-value", value=0.05, min=0.001, max=0.1, step=0.01),
                        ui.input_action_button("compute_volcano", "Compute", class_="btn-primary"),
                        width=1/4
                    ),
                    ui.output_ui("volcano_plot_content"), 
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("🎓 What is a Volcano Plot?"),
                    ui.div(ui.output_ui("volcano_explanation_ui"), style="max-height: 600px; overflow-y: auto;")
                ),
                width=1/2, heights_equal="row"
            )
        else:
            return ui.card(
                ui.card_header("Volcano Plot"),
                ui.layout_column_wrap(
                    ui.tooltip(
                        ui.input_select(
                            "volcano_var", 
                            "Variable", 
                            choices=['er_status','type_of_breast_surgery','chemotherapy',
                                    'hormone_therapy','radio_therapy','primary_tumor_laterality',
                                    'er_status_measured_by_ihc','pr_status','her2_status']
                        ),
                        "Binary clinical variable to compare groups. Genes will be tested for differential expression between the two groups.",
                        placement="top"
                    ),
                    
                    ui.tooltip(
                        ui.input_numeric("fc_threshold", "FC Threshold", value=1.0, min=0.1, max=3.0, step=0.1),
                        "Minimum log₂ fold-change to consider a gene significant.",
                        placement="top"
                    ),
                    
                    ui.tooltip(
                        ui.input_numeric("p_threshold", "P-value", value=0.05, min=0.001, max=0.1, step=0.01),
                        "Maximum p-value threshold. Lower values (e.g., 0.01) are more stringent and reduce false positives.",
                        placement="top"
                    ),
                    
                    ui.input_action_button("compute_volcano", "Run", class_="btn-primary"),
                    
                    width=1/4, 
                    heights_equal="row"
                ),
                ui.output_ui("volcano_plot_content"), 
                full_screen=True
            )

    # ========================================================================
    # DIMENSIONALITY REDUCTION
    # ========================================================================
    dr_results = reactive.Value(None)
    dr_runs = reactive.Value([])
    recent_dr_plots = reactive.Value([])
    intrinsic_dim_data = reactive.Value(None)

    @render.ui
    def dr_tabs_dynamic():
        tabs = [
            ui.nav_panel("Intrinsic Dimensionality", ui.output_ui("intrinsic_dim_layout", height="500px")),
            ui.nav_panel("Projection", ui.input_selectize("dr_color_by", "Color By:", choices=get_groupby_choices(), selected="er_status"), ui.output_ui("projection_plots", height="500px")),
            ui.nav_panel("Neighborhood Preservation", ui.output_ui("neighborhood_preservation_layout")),
            ui.nav_panel(
                "Past Runs", 
                ui.card(ui.card_header("Run History"), ui.output_table("dr_runs_table"),full_screen=True, style="height: 300px; overflow-y: auto;"),
                ui.br(),
                ui.card( ui.output_ui("dr_comparison_plots"), full_screen=True, style="min-height: 500px;")
            )
        ]
        
        if active_mode.get() == "learning":
            guide_tab = ui.nav_panel(
                "🎓 Learner Guide",
                ui.markdown("""
                    The METABRIC dataset contains **489 gene expression variables** per patient. 
                    Visualizing data in 489 dimensions is impossible for humans.

                    Dimensionality Reduction (DR) techniques allow us to have a lower-dimensional representation (an embedding), typically 2D or 3D for visualization.

                    Ultimately, we hope that embedding preserves most of the high dimensional genomic structure to make inferences.<br><br>

                    ##### **Why reduce dimensions?**
                    
                    - **Visualization**: Humans can only see 2D/3D, not 
                            
                    - **Pattern Discovery**: Hidden relationships become visible
                    - **Noise Reduction**: Focus on the most important variations
                    - **Computational Efficiency**: Speeds up downstream analysis (clustering, etc.)
                    <br><br>
                    
                    ##### **Understanding the tabs**
                    
                    1. **Intrinsic Dimensionality**: How many dimensions does the data really have?
                    
                        This translates the idea that the genomic data might lie on a lower-dimensional structure (see [Manifold Hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis))
                        Tow techines are used to estimate the intrinsic dimensionality : (i) [Exponential Correlation](https://link.springer.com/chapter/10.1007/978-0-387-21830-4_12) and (ii) [Fisher separability](https://arxiv.org/abs/1901.06328).<br><br>
                    
                    2. **Projection**: How does our data look in 2D/3D?
                       
                        We allow you to play with 2 dimensionality reduction techinques :
                            
                        - [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) : a fast, interpretable, parameter-free method that finds a lower dimension of so called *principal components*, that find the direction of the maximum variance.
                        - [UMAP](https://pair-code.github.io/understanding-umap/) : a more sophisticated non-linear technique. <br><br>

                        PCA is typically good for data with linear relationships, and for initial exploration. UMAP on the other hand can be better for grasping more complex (non-linear) structures. There are two main parameters at play : (i) **N Neighbors**, low values emphasizes local structures while high preserves local ones ; (ii) **Min distance**, the higher you set that parameters, the more the projected points tend to "spread out' (click [here](https://pair-code.github.io/understanding-umap/) for an intuitive exploration of the parameters).<br><br>

                    3. **Neighborhood Preservation**: How much of the initial structure do we preserve?
                       For each metric, we compute its value for different neighborhood sizes K (from 1 to the sample size you selected). This produces a **curve** showing how well the projection preserves structure at different scales.<br>
                        1. **Trustworthiness** T : Quantifies whether the embedding preserves local relationships found in the original data.

                        2. **Continuity** C : Quantifies how accurately the embedding reflects the original structure of the data.

                        3. **Behavior** B_NX: Quantifies the tendency of the reduction to favor either extrusive or intrusive behaviors. A positive value indicates that we have more mild intrusions than extrusions, while a negative value tells the opposite.

                        4. **Normalized Quality** R_NX: Measures the proportion of correctly ranked neighbors up to size K, producing a curve that reveals whether the projection emphasizes local structure (high values at low K) or global relationships (plateau across all K). It is often preferred over the unscaled Quality, as Normalized Quality yields a value of 0 for a random embedding, unlike unscaled Quality.

                        5. **Area Under the Curve** AUC : Provides the area under the curve of the R_NX. In other words instead of a sequence R_NX(k) k={1,...,K}, we have a number between 0 ans 1 that summerizes the sequence (pretty cool!) <br><br>
                            
                        **How to interprete those ?** Very straighforward, we want each point in the sequences C, T, R_NX, to be close to 1, as for the AUC. And we want all points in B_NX sequence close to 0.
                """)
            )
            tabs.insert(0, guide_tab)
        
        return ui.navset_card_pill(*tabs, id="dr_tabs")

    def store_run(params):
        current = dr_runs.get()
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        results = dr_results.get()
        preservation_df = results['preservation']['metrics_df']
        
        auc_metric = input.auc_base_metric() if active_mode.get() == "expert" else "R_NX"
        auc_umap = compute_auc_metric(preservation_df, metric_name=auc_metric, method='UMAP')
        params["AUC_UMAP"] = f"{auc_umap:.3f}" if not np.isnan(auc_umap) else "N/A"
        
        if input.include_pca():
            auc_pca = compute_auc_metric(preservation_df, metric_name=auc_metric, method='PCA')
            params["AUC_PCA"] = f"{auc_pca:.3f}" if not np.isnan(auc_pca) else "N/A"
        
        params["Timestamp"] = timestamp
        dr_runs.set(current + [params])
        
        n_dims = results['n_components']
        umap_emb = results['umap']
        color_by = input.dr_color_by() if input.dr_color_by() != "none" else None
        df_umap = create_projection_df(embedding=umap_emb, df_original=df, sample_indices=results['sample_indices'], color_var=color_by, method="UMAP")
        
        if n_dims == 2:
            fig = plot_bivariate(df_umap, x_var='UMAP_1', y_var='UMAP_2', color_by=color_by)
        else:
            fig = px.scatter_3d(df_umap, x='UMAP_1', y='UMAP_2', z='UMAP_3', color=color_by, template='plotly_white', color_discrete_sequence=OKABE_ITO_COLORS)
        
        current_plots = recent_dr_plots.get()
        plot_info = {
            "fig": fig,
            "title": f"UMAP {n_dims}D | n={input.umap_n_neighbors()} | d={input.umap_min_dist()}",
            "time": timestamp
        }
        recent_dr_plots.set([plot_info] + current_plots[:2])  

    @reactive.Effect
    @reactive.event(input.run_dr)
    def compute_dr():        
        start_DR = time.time()
        n_steps = 3
        n = 1
        
        with ui.Progress() as p:
            if input.include_pca():
                n_steps += 1
            
            # 1. Intrinsic Dimensionality (Computed once)
            if intrinsic_dim_data.get() is None:
                n_steps += 1
                p.set(message=f"[{n}/{n_steps}] Computing intrinsic dimensionality...", detail="The intrinsic dimensionality is computed on all the genomic profiles. But don't worry, it is only ran once!")
                n += 1
                X_full, _, _ = prepare_feature_matrix(df=df, gene_cols=GENE_EXPRESSION_VARIABLES, sample_fraction=1.0)
                intrinsic_dim_data.set(estimate_all_dimensions(X_full))

            # 2. Data Preparation
            p.set(message=f"[{n}/{n_steps}] Prepare feature matrix...", detail=f"Using {input.dr_sample_fraction()}% of data")
            n += 1
            X, sample_indices, n_features = prepare_feature_matrix(df=df, gene_cols=GENE_EXPRESSION_VARIABLES, sample_fraction=input.dr_sample_fraction() / 100)
            
            n_components = int(input.dr_n_components())
            results = {'X_original': X, 'sample_indices': sample_indices, 'n_features': n_features, 'n_components': n_components}
            
            # 3. UMAP Calculation
            p.set(message=f"[{n}/{n_steps}] Computing UMAP ({n_components}D)...",  detail="This may take a bit time. If it's too long for you, consider diminushing the sample size !" if input.dr_sample_fraction()>10 else "This should be quick enough, you only selected 10% of the data!")
            n += 1
            umap_emb = compute_umap(
                X, n_components=n_components, n_neighbors=input.umap_n_neighbors(), min_dist=input.umap_min_dist(),
                metric=input.umap_metric() if active_mode.get() == "expert" else "euclidean",
                random_state=input.umap_random_state() if active_mode.get() == "expert" else 2003
            )
            results['umap'] = umap_emb
            
            # 4. PCA Calculation
            if input.include_pca():
                p.set(message=f"[{n}/{n_steps}] Computing PCA (20D)...")
                n += 1
                pca_emb, pca_var = compute_pca(X, n_components=20)
                results['pca'] = {'embedding': pca_emb, 'variance_explained': pca_var}
                
                p.set(message=f"[{n}/{n_steps}] Evaluating neighborhood preservation...", detail=f"Computing Trustworthiness, Continuity, Normalized Quality and behaviour for PCA and UMAP (K_max = {X.shape[0]})")
                n += 1
                preservation_res = compute_neighborhood_preservation(X_high=X, X_low_umap=umap_emb, X_low_pca=pca_emb[:, :n_components], K_values=X.shape[0])
                results['preservation'] = preservation_res
                
                # Calculate summary stats for plot titles
                pca_var_sum = np.sum(pca_var[:n_components]) * 100
                metrics_df = preservation_res['metrics_df']
                umap_auc = compute_auc_metric(metrics_df, metric_name='R_NX', method='UMAP')
                
                results['metrics_summary'] = {
                    'pca_var': f"{pca_var_sum:.1f}%",
                    'umap_auc': f"{umap_auc:.3f}"
                }

            else:
                p.set(message=f"[{n}/{n_steps}] Evaluating neighborhood preservation...", detail=f"Computing Trustworthiness, Continuity, Normalized Quality and behaviour for UMAP (K_max = {X.shape[0]})")
                n += 1
                preservation_res = compute_neighborhood_preservation(X_high=X, X_low_umap=umap_emb, X_low_pca=None, K_values=X.shape[0])
                results['preservation'] = preservation_res
                
                # --- METRICS SUMMARY ---
                metrics_df = preservation_res['metrics_df']
                umap_auc = compute_auc_metric(metrics_df, metric_name='R_NX', method='UMAP')
                
                results['metrics_summary'] = {
                    'umap_auc': f"{umap_auc:.3f}",
                    'pca_var': "N/A"
                }
        
        dr_results.set(results)
        
        store_run({
            "Sample %": input.dr_sample_fraction(),
            "N components": int(input.dr_n_components()),
            "N neighbors": input.umap_n_neighbors(),
            "Min dist": input.umap_min_dist(),
            "Include PCA": input.include_pca(),
            **({
                "Metric": input.umap_metric(),
                "Random State": input.umap_random_state()
            } if active_mode.get() == "expert" else {})
        })
        
        ui.notification_show(f"Done !", type="success")

        end_DR = time.time()
        print(f"DIMENSIONALITY REDUCTION TOOK {end_DR-start_DR:.2f} seconds")
        dr_results.set(results)
        store_run({
            "Sample %": input.dr_sample_fraction(),
            "N components": int(input.dr_n_components()),
            "N neighbors": input.umap_n_neighbors(),
            "Min dist": input.umap_min_dist(),
            "Include PCA": input.include_pca(),
            **({
                "Metric": input.umap_metric(),
                "Random State": input.umap_random_state()
            } if active_mode.get() == "expert" else {})
        })
        ui.notification_show(f"Done !", type="success")

        end_DR = time.time()
        print(f"DIMENSIONALITY REDUCTION TOOK {end_DR-start_DR}")

    @render.ui
    def correlation_dim_plot():
        dim_data = intrinsic_dim_data.get()
        if dim_data is None or dim_data.empty:
            return ui.div(
                ui.markdown("Choose your parameters and click the **Run** button in the sidebar to see results"),
                style="text-align: center; min-height: 400px; display: flex; align-items: center; justify-content: center;"
            )
        return output_widget("correlation_dim_widget")

    @render_widget
    def correlation_dim_widget():
        return plot_correlation_dim(intrinsic_dim_data.get())

    @render_widget
    def scree_plot():
        results = dr_results.get()
        if results is None or 'pca' not in results:
            return None
        return plot_scree(results['pca']['variance_explained'])
    
    @render.ui
    def intrinsic_dim_layout():
        include_pca = input.include_pca()
        if not include_pca:
            return ui.card(ui.card_header("Ad Hoc Estimation"), ui.output_ui("correlation_dim_plot"), full_screen=True)
        else:
            return ui.layout_column_wrap(
                ui.card(ui.card_header("Ad Hoc Estimation"), ui.output_ui("correlation_dim_plot"), full_screen=True),
                ui.card(ui.card_header("Variance Explained (PCA)"), output_widget("scree_plot"), full_screen=True),
                width=1/2, heights_equal="row"
            )

    @render.ui
    def projection_plots():
        results = dr_results.get()
        include_pca = input.include_pca()
        
        if not results:
            return ui.div(
                ui.markdown("Choose your parameters and click the **Run** button in the sidebar to see results"),
                style="text-align: center; min-height: 400px; display: flex; align-items: center; justify-content: center;"
            )
        
        if 'preservation' in results and 'metrics_df' in results['preservation']:
            metrics_df = results['preservation']['metrics_df']
            auc_umap = compute_auc_metric(metrics_df, metric_name='R_NX', method='UMAP')
            title_umap = f"UMAP Projection (Quality AUC: {auc_umap:.3f})"
        else:
            title_umap = "UMAP Projection"

        if include_pca and 'pca' in results:
            var_explained = np.sum(results['pca']['variance_explained']) * 100
            title_pca = f"PCA Projection (Variance: {var_explained:.1f}%)"
        else:
            title_pca = "PCA Projection"
        
        if not include_pca:
            return ui.card(ui.card_header(title_umap), output_widget("umap_plot"), full_screen=True)
        else:
            return ui.layout_columns(
                ui.card(ui.card_header(title_umap), output_widget("umap_plot"), full_screen=True),
                ui.card(ui.card_header(title_pca), output_widget("pca_plot"), full_screen=True),
                col_widths=(6, 6)
            )

    @render_widget
    def umap_plot():
        results = dr_results.get()
        color_by = input.dr_color_by() if input.dr_color_by() != "none" else None
        df_umap = create_projection_df(embedding=results['umap'], df_original=df, sample_indices=results['sample_indices'], color_var=color_by, method="UMAP")
        n_dims = results['n_components']
        if n_dims == 2:
            return plot_bivariate(df_umap, x_var='UMAP_1', y_var='UMAP_2', color_by=color_by)
        else:
            return px.scatter_3d(df_umap, x='UMAP_1', y='UMAP_2', z='UMAP_3', color=color_by, template='plotly_white', color_discrete_sequence=OKABE_ITO_COLORS)

    @render_widget
    def pca_plot():
        results = dr_results.get()
        n_dims = results['n_components']
        pca_emb = results['pca']['embedding'][:, :n_dims]
        color_by = input.dr_color_by() if input.dr_color_by() != "none" else None
        df_pca = create_projection_df(embedding=pca_emb, df_original=df, sample_indices=results['sample_indices'], color_var=color_by, method="PCA")
        if n_dims == 2:
            return plot_bivariate(df_pca, x_var='PCA_1', y_var='PCA_2', color_by=color_by)
        else:
            return px.scatter_3d(df_pca, x='PCA_1', y='PCA_2', z='PCA_3', color=color_by, template='plotly_white', color_discrete_sequence=OKABE_ITO_COLORS)

    @render.ui
    def neighborhood_preservation_layout():
        results = dr_results.get()
        if results is None or 'preservation' not in results:
            return ui.div(
                ui.markdown("Choose your parameters and click the **Run** button in the sidebar to see results"),
                style="text-align: center; min-height: 500px; display: flex; align-items: center; justify-content: center;"
            )
        return ui.card(output_widget("quality_metrics_plot"), full_screen=True)

    @render_widget
    def quality_metrics_plot():
        results = dr_results.get()
        if results is None or 'preservation' not in results:
            return None
        return plot_quality_metrics(results['preservation']['metrics_df'])

    @render.table(index=False)
    def dr_runs_table():
        runs = dr_runs.get()
        if len(runs) == 0:
            return pd.DataFrame({"Info": ["No dimensionality reduction runs yet"]})
        return pd.DataFrame(runs).style.set_table_attributes('class="dataframe shiny-table table w-auto"')
    
    @render.ui
    def dr_comparison_plots():
        plots_data = recent_dr_plots.get()
        
        if not plots_data:
            return ui.div(
                ui.markdown("No runs yet. Run a dimensionality reduction to see comparisons here."),
                style="text-align: center; color: #6c757d; padding: 50px;"
            )
        
        return ui.layout_column_wrap(
            ui.card(ui.card_header("Run 1 (Newest)"), output_widget("dr_comp_plot_1")),
            ui.card(ui.card_header("Run 2"), output_widget("dr_comp_plot_2")),
            ui.card(ui.card_header("Run 3 (Oldest)"), output_widget("dr_comp_plot_3")),
            width=1/3
        )

    @render_widget
    def dr_comp_plot_1():
        plots = recent_dr_plots.get()
        return plots[0]['fig'] if len(plots) > 0 else None

    @render_widget
    def dr_comp_plot_2():
        plots = recent_dr_plots.get()
        return plots[1]['fig'] if len(plots) > 1 else None

    @render_widget
    def dr_comp_plot_3():
        plots = recent_dr_plots.get()
        return plots[2]['fig'] if len(plots) > 2 else None


    #==========================================================================================
    # BRIDGING (Parameters form DR to clustering preprocessing)
    #==========================================================================================
    
    @reactive.Effect
    @reactive.event(input.import_dr_settings)
    def import_dr_parameters():
        last_dr = dr_results.get()
        if last_dr is None:
            ui.notification_show("No Dimensionality Reduction run found!", type="warning")
            return
       
        ui.notification_show("Importing UMAP parameters from DR tab...", type="message")
       
        ui.update_select("clust_prep_method", selected="umap")
        ui.update_slider("clust_prep_umap_neighbors", value=input.umap_n_neighbors())
        ui.update_slider("clust_prep_umap_mindist", value=input.umap_min_dist())
       
        ui.update_select("db_prep_method", selected="umap")
        ui.update_slider("db_prep_umap_neighbors", value=input.umap_n_neighbors())
        ui.update_slider("db_prep_umap_mindist", value=input.umap_min_dist())
       
        ui.update_select("hc_prep_method", selected="umap")
        ui.update_slider("hc_prep_umap_neighbors", value=input.umap_n_neighbors())
        ui.update_slider("hc_prep_umap_mindist", value=input.umap_min_dist())

    def transfer_dr_settings(prefix):
        n_neighbors = input.umap_n_neighbors()
        min_dist = input.umap_min_dist()
       
        n_dims = int(input.dr_n_components())
       
        ui.notification_show(f"Importing UMAP settings ({n_dims}D, n={n_neighbors}, d={min_dist}) to {prefix.upper()}...", type="message", duration=3)
       
       
        ui.update_select(f"{prefix}_prep_method", selected="umap")
        ui.update_slider(f"{prefix}_prep_umap_neighbors", value=n_neighbors)
        ui.update_slider(f"{prefix}_prep_umap_mindist", value=min_dist)
        ui.update_numeric(f"{prefix}_prep_umap_n", value=n_dims)
        ui.update_radio_buttons(f"{prefix}_proj_dims", selected=str(n_dims))
   
    @reactive.Effect
    @reactive.event(input.import_dr_kmeans)
    def _():
        transfer_dr_settings("clust")

    @reactive.Effect
    @reactive.event(input.import_dr_hc)
    def _():
        transfer_dr_settings("hc")

    @reactive.Effect
    @reactive.event(input.import_dr_db)
    def _():
        transfer_dr_settings("db")
    # ========================================================================
    # CLUSTERING - HISTORY LOGGING
    # ========================================================================
    cluster_history = reactive.Value([])

    @render.ui
    def kmeans_tabs_dynamic():
        res = cluster_results.get()
        if res is None:
            placeholder = ui.div(
                ui.markdown("Choose your parameters and click the **Run** button in the sidebar to see results"),
                style="text-align: center; min-height: 500px; display: flex; align-items: center; justify-content: center;"
            )
            content_map = placeholder
            content_uni = placeholder
            content_bi = placeholder
        else:
            content_map = ui.layout_columns(
                ui.card(output_widget("cluster_scatter_plot"), full_screen=True),
                ui.layout_column_wrap(
                    ui.card(ui.card_header("Quality Metrics"), ui.output_table("cluster_metrics_table")),
                    ui.card(ui.card_header("Optimal k (CH & Gap)"), output_widget("cluster_metrics_plot")),
                    width=1, heights_equal="width"
                ),
                col_widths=(8, 4)
            )
           
            content_uni = ui.card(
                ui.layout_column_wrap(
                    ui.input_selectize("clust_uni_var", "Main Variable:", choices=get_clinical_choices(), selected="age_at_diagnosis"),
                    ui.input_selectize("clust_uni_subgroup", "Sub-group by:", choices=get_groupby_choices(), selected="none"),
                    width=1/2
                ),
                output_widget("cluster_univariate_plot"),
                full_screen=True
            )
           
            content_bi = ui.card(ui.card_header("Cluster vs Gene Expression"), output_widget("cluster_heatmap"), full_screen=True)

        tabs = [
            ui.nav_panel("Cluster Map", content_map),
            ui.nav_panel("Univariate", content_uni),
        ]

        if active_mode.get() == "learning":
            guide_tab = ui.nav_panel(
                "🎓 Learner Guide",
                ui.markdown(r"""
                The **K-mean** algorithm is a clustering technique where patient are separated into k distinct groups to minimise the difference within each group.
                <br>
                * It places k centers randomly.
                * It assigns each patient to the closest center.
                * It moves the center to the average position of its patients.
                * Repeat until the centroid are stable.               

                Most of the time when we perform a K-means with high dimensional data, we perform at first a DR technique.
                The goal is to avoid the problem of "Curse of Dimensionality" (see [Surprising Behavior of Distance Metric](https://www.researchgate.net/publication/30013021_On_the_Surprising_Behavior_of_Distance_Metric_in_High-Dimensional_Space)). With 489 genes, the space is so big that the distances between observations become
                meaningless and K-means could have some problems to perform and also the result could be biais.
                <br>
                That is why we advise the user to reduce to correct dimension first, so that we can ensure that K-means groups patients based on distance between individual rather than noise.
                <br>
                <br>
                           
                In order to find the optimal k, we advise the user to look at two differents metrics.
           
                **Calinski-Harabasz (CH)** ([Communications in Staticis](https://www.researchgate.net/profile/Tadeusz-Calinski/publication/233096619_A_Dendrite_Method_for_Cluster_Analysis/links/555213e108aeaaff3befe29b/A-Dendrite-Method-for-Cluster-Analysis.pdf)) : Measures how distinct the clusters are.          
                **Gap Statistic** ([Estimating the number of clusters](https://www.jstor.org/stable/2680607?seq=1)): Compares your clustering to a random distribution.
                """)
            )
            tabs.insert(0, guide_tab)

        return ui.navset_card_pill(*tabs, id="kmeans_tabs")
   

    def log_run(algo, prep, params, k, ch, gap, other_metric, viz, fig=None):
        #Historic register
        current_history = cluster_history.get()
        new_entry = {
            "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
            "Algorithm": algo,
            "Preprocessing": prep,
            "Params/Metric": params,
            "Clusters (k)": k,
            "CH Score": ch,
            "Gap Stat": gap,
            "Sil./DB/DBCV": other_metric,
            "Visualization": viz
        }
        cluster_history.set([new_entry] + current_history)

        if fig is not None: #if coud be none if the clustering technique doesn't find different cluster -->DBScan
            current_plots = recent_plots.get()
            plot_info = {
                "fig": fig,
                "title": f"{algo} | k={k} | {prep}",
                "time": new_entry["Timestamp"]
            }
            new_list = [plot_info] + current_plots
            recent_plots.set(new_list[:3])

    @render.table
    def cluster_history_table():
        history = cluster_history.get()
        if not history:
            return pd.DataFrame({"Status": ["No runs yet. Run an analysis to see history here."]})
        return pd.DataFrame(history).style.set_table_attributes('class="table table-striped table-hover"').set_properties(**{'text-align': 'center'})

    # ========================================================================
    # K-MEANS LOGIC
    # ========================================================================

#We put a lot of try and except to avoid the crash of the app --> Prefers to get an erro message

    cluster_results = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.run_clustering)
    def compute_clustering(): #Compute k-means
        with ui.Progress() as p:

            p.set(message="[1/5] Preparing data...", detail="Subsampling...")
            X_raw, sample_indices, _ = prepare_feature_matrix(df=df, gene_cols=GENE_EXPRESSION_VARIABLES, sample_fraction=input.clust_sample_fraction() / 100)
            
            info_prep = "None"
            prep_method = input.clust_prep_method()
            
            if prep_method == "pca":
                n_keep = input.clust_prep_pca_n()
                p.set(message="[2/5] Preprocessing...", detail=f"PCA ({n_keep} dims)")
                X_prep, var_ratio = compute_pca(X_raw, n_components=n_keep)
                total_var = np.sum(var_ratio) * 100
                info_prep = f"PCA ({n_keep}D): {total_var:.1f}% Var."

            elif prep_method == "umap":
                n_keep = input.clust_prep_umap_n()
                n_neighbors = input.clust_prep_umap_neighbors()
                min_dist = input.clust_prep_umap_mindist()
                p.set(message="[2/5] Preprocessing...", detail=f"UMAP ({n_keep} dims)")
                
                X_prep = compute_umap(X_raw, n_components=n_keep, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                
                try:
                    k_neigh = min(15, len(X_prep) - 1)
                    if k_neigh > 0:
                        score_prep = trustworthiness(X_raw, X_prep, n_neighbors=k_neigh)
                        info_prep = f"UMAP ({n_keep}D): {score_prep:.2f} (Trust)"
                    else:
                        info_prep = f"UMAP ({n_keep}D): Non-linear"
                except:
                    info_prep = f"UMAP ({n_keep}D): Non-linear"

            else:
                p.set(message="[2/5] Preprocessing...", detail="Raw Data")
                X_prep = X_raw
                info_prep = "Raw Data (489 Genes)"

            with parallel_backend('threading'):
                k_sel = input.clust_k()
                p.set(message="[3/5] Clustering...", detail=f"K-Means (k={k_sel})")
                model = KMeans(n_clusters=k_sel, n_init=input.clust_n_init(), random_state=42)
                labels = model.fit_predict(X_prep)

                # Metrics calculation
                wcss = model.inertia_
                X_mean = X_prep.mean(axis=0)
                tss = np.sum((X_prep - X_mean) ** 2)
                r2_score = 1 - (wcss / tss)
                db_score = davies_bouldin_score(X_prep, labels)

                # Elbow / Gap calculation
                p.set(message="4/5 Computing Elbow...", detail="Testing k=2 to 20")
                ks = list(range(2, 21))
                ch, gap = [], []
                min_v, max_v = np.min(X_prep, axis=0), np.max(X_prep, axis=0)
                for k in ks:
                    km = KMeans(k, n_init=3, random_state=42).fit(X_prep)
                    ch.append(calinski_harabasz_score(X_prep, km.labels_))
                    obs = np.log(km.inertia_)
                    refs = [np.log(KMeans(k, n_init=1).fit(np.random.uniform(min_v, max_v, X_prep.shape)).inertia_) for _ in range(3)]
                    gap.append(np.mean(refs) - obs)

            # Visualization
            p.set(message="5/5 Visualization...", detail="Projecting coordinates")
            
            n_dims_current = X_prep.shape[1]
            v_inf = "Raw Data"
            
            if n_dims_current <= 3:
                emb = X_prep
                if prep_method == 'pca':
                    cols = [f'PC{i+1}' for i in range(n_dims_current)]
                    v_inf = f"PCA ({n_dims_current}D): {total_var:.1f}% Var." 
                elif prep_method == 'umap':
                    cols = [f'UMAP{i+1}' for i in range(n_dims_current)]
                    v_inf = info_prep
                else:
                    cols = [f'Dim{i+1}' for i in range(n_dims_current)]
            else:
                viz = input.clust_viz_method()
                dim = int(input.clust_proj_dims())
                
                if viz == 'pca':
                    emb, var_ratio_viz = compute_pca(X_prep, n_components=dim)
                    total_var_viz = np.sum(var_ratio_viz) * 100
                    cols = [f'PC{i+1}' for i in range(dim)]
                    v_inf = f"PCA ({dim}D): {total_var_viz:.1f}% Var."
                else:
                    viz_neighbors = input.clust_viz_umap_neighbors()
                    viz_min_dist = input.clust_viz_umap_min_dist()
                    emb = compute_umap(X_prep, n_components=dim, n_neighbors=viz_neighbors, min_dist=viz_min_dist)
                    cols = [f'UMAP{i+1}' for i in range(dim)]
                    
                    try:
                        k_neigh = min(15, len(X_prep) - 1)
                        if k_neigh > 0:
                            score = trustworthiness(X_prep, emb, n_neighbors=k_neigh)
                            v_inf = f"UMAP ({dim}D): {score:.2f} (Trust)"
                        else:
                            v_inf = f"UMAP ({dim}D): Non-linear"
                    except Exception as e:
                        print(f"Erreur Metric Viz: {e}")
                        v_inf = f"UMAP ({dim}D): Non-linear"

            # Extraction CH/Gap for the corresponded k
            try:
                idx_k = ks.index(k_sel)
                cur_ch = ch[idx_k]
                cur_gap = gap[idx_k]
            except ValueError:
                cur_ch, cur_gap = 0, 0

            cluster_results.set({
                'labels': labels, 'embedding': emb, 'embedding_cols': cols, 'sample_indices': sample_indices, 'k': k_sel,
                'metrics': {'wcss': wcss, 'r2': r2_score, 'davies_bouldin': db_score, 'info_prep': info_prep, 'info_viz': v_inf, 'ch': cur_ch, 'gap': cur_gap},
                'evolution': {'ks': ks, 'ch': ch, 'gap': gap, 'best_ch': ks[np.argmax(ch)], 'best_gap': ks[np.argmax(gap)]}
            })
            
            final_fig = plot_cluster_map(emb, cols, labels, df.iloc[sample_indices])
            log_run("K-Means", info_prep, "Euclidean", k_sel, f"{cur_ch:.1f}", f"{cur_gap:.3f}", f"DB:{db_score:.2f}", v_inf, fig=final_fig)
            ui.notification_show("K-Means complete!", type="success")

    @render.table
    def cluster_metrics_table():
        res = cluster_results.get()
        if res is None: return None
        
        m = res['metrics']
        df = pd.DataFrame({
            "M": ["Preprocessing", "CH Score (Separation)", "Gap Statistic", "Visualization"],
            "V": [m['info_prep'], f"{m['ch']:.1f}", f"{m['gap']:.3f}", m['info_viz']]
        })
        df.columns = ["", ""] #Hide the title of the columns
        return df
   
    @render_widget
    def cluster_metrics_plot():
        res = cluster_results.get()
        if not res:
            return None
        e = res['evolution']
        return plot_kmeans_metrics(e['ks'], e['ch'], e['gap'], res['k'], e['best_ch'], e['best_gap'])

    @render_widget
    def cluster_scatter_plot():
        res = cluster_results.get()
        if not res:
            return None
        return plot_cluster_map(res['embedding'], res['embedding_cols'], res['labels'], df.iloc[res['sample_indices']])

    @render_widget
    def cluster_univariate_plot():
        res = cluster_results.get()
        if not res: return None
        try:
            var = input.clust_uni_var()
            sub = input.clust_uni_subgroup()
        except:
            return None

        return plot_cluster_univariate(df.iloc[res['sample_indices']], res['labels'], var, sub, format_label)
    
    # ========================================================================
    # HIERARCHICAL LOGIC
    # ========================================================================
    hc_results = reactive.Value(None)

    @render.ui
    def hc_tabs_dynamic():
        res = hc_results.get()
       
        if res is None:
            placeholder = ui.div(#Try to first put a placeholder function, but it was easy to put it by hand 
                ui.markdown("Choose your parameters and click the **Run** button in the sidebar to see results"),
                style="text-align: center; min-height: 500px; display: flex; align-items: center; justify-content: center;"
            )
            content_map = placeholder
            content_uni = placeholder
        else:
            content_map = ui.TagList(
                ui.layout_columns(
                    ui.card(ui.card_header("Cluster Visualization"), output_widget("hc_scatter_plot"), full_screen=True, height="600px"),
                    ui.layout_column_wrap(
                        ui.card(ui.card_header("Metrics"), ui.output_table("hc_metrics_table"), height="auto"),
                        ui.card(ui.card_header("Optimal k (CH & Gap)"), output_widget("hc_metrics_plot"), height="400px"),
                        width=1, heights_equal="min"
                    ),
                    col_widths=(8, 4)
                ),
                ui.br(),
                ui.card(ui.card_header("Dendrogram"), ui.output_plot("hc_dendrogram_plot"), full_screen=True, height="500px")
            )
           
            content_uni = ui.card(
                ui.layout_column_wrap(
                    ui.input_selectize("hc_uni_var", "Main Variable:", choices=get_clinical_choices(), selected="age_at_diagnosis"),
                    ui.input_selectize("hc_uni_subgroup", "Sub-group by:", choices=get_groupby_choices(), selected="none"),
                    width=1/2
                ),
                output_widget("hc_univariate_plot"),
                full_screen=True
            )

        tabs = [
            ui.nav_panel("Cluster Map", content_map),
            ui.nav_panel("Univariate", content_uni)
        ]

        if active_mode.get() == "learning":
            guide_tab = ui.nav_panel(
                "🎓 Learner Guide",
                ui.markdown(r"""
                Hierarchical Clustering does not require specifying the number of clusters before runing it. Instead, it builds a **hierarchy** of clusters that can be visualized as a tree (Dendrogram).
                <br>
                In the sidebar, the user can choose a number of clusters (k), but this is primarily used to visualize the resulting groups in the scatter plot of clustered individuals.

                The hierarchical clustering method used in this dashboard is the **Bottom-Up** approach:
                1.  Treat each patient as a single cluster.
                2.  Identify the two closest clusters and merge them into one.
                3.  Continue merging until all patients belong to a single giant cluster.

                Just like with K-Means, calculating the "distance" between patients is the core of this algorithm.
                <br>
                With 489 genes (high-dimensional space), the **Curse of Dimensionality** makes all patients appear equally distant. Reducing dimensions first restores meaningful distances and makes the resulting tree biologically relevant.

                The **linkage methods** used to compute the distance between two clusters are Centroid linkage and Minimax linkage. These are two methods commonly used in bioinformatics.

                * **Centroid linkage**: Measures the distance between the averages (centers) of two groups.
                * **Complete Linkage**: Measures the maximum distance between any pair of points in two different clusters.
                """)
            )
            tabs.insert(0, guide_tab)

        return ui.navset_card_pill(*tabs, id="hc_tabs")
       
    def calculate_wcss_hc(X, labels):#Within sum of square for HC
        wcss = 0
        unique_labs = np.unique(labels)
        for i in unique_labs:
            points = X[labels == i]
            if len(points) > 1:
                center = np.mean(points, axis=0)
                wcss += np.sum((points - center) ** 2)
        return wcss
    
    @reactive.Effect
    @reactive.event(input.run_hc)
    def compute_hierarchical():#As said before, we had a lot of crashes, so we prefered to used a lot of try/except to avoid crashes
        with ui.Progress() as p:
            info_prep = "Processing..."
            info_viz = "Processing..."
            curr_ch, curr_gap = 0, 0
            
            try:
                p.set(message="[1/4] Processing...", detail="Data Reduction")
                frac = input.hc_sample_fraction() / 100.0
                X_raw, sample_indices, _ = prepare_feature_matrix(df, GENE_EXPRESSION_VARIABLES, frac)
                
                n_samples = len(X_raw)
                if n_samples < 5:
                    ui.notification_show("Not enough data (<5 samples).", type="error")
                    return

                method_prep = input.hc_prep_method()

                if method_prep == "pca":
                    safe_n = min(input.hc_prep_pca_n(), n_samples, X_raw.shape[1])
                    pca_mod = PCA(n_components=safe_n)
                    X_use = pca_mod.fit_transform(X_raw)
                    var_pct = np.sum(pca_mod.explained_variance_ratio_)*100
                    info_prep = f"PCA ({safe_n}D): {var_pct:.1f}% Var."
                elif method_prep == "umap":
                    n_keep = input.hc_prep_umap_n()
                    req_neighbors = input.hc_prep_umap_neighbors()
                    safe_neighbors = min(req_neighbors, n_samples - 2)
                    if safe_neighbors < 2: safe_neighbors = 2
                    X_use = compute_umap(X_raw, n_components=n_keep, n_neighbors=safe_neighbors, min_dist=input.hc_prep_umap_mindist())
                    try:
                        k_neigh = min(15, n_samples - 1)
                        score = trustworthiness(X_raw, X_use, n_neighbors=k_neigh)
                        info_prep = f"UMAP ({n_keep}D): {score:.2f} (Trust)"
                    except:
                        info_prep = f"UMAP ({n_keep}D): Non-linear"
                else:
                    X_use = X_raw
                    info_prep = "Raw Data"

                #  Clustering (Linkage Real)
                p.set(message="[2/4] Building Tree...", detail="Real Data Linkage")
                link = input.hc_linkage()
                sys.setrecursionlimit(50000)
                Z = linkage(X_use, method=link, metric='euclidean')

                # Optimised gat statistic
                p.set(message="[2.5/4] Gap Statistic...", detail="Reference Linkage")
                
                mins = np.min(X_use, axis=0)
                maxs = np.max(X_use, axis=0)
                X_ref = np.random.uniform(mins, maxs, X_use.shape)
                
                Z_ref = linkage(X_ref, method=link, metric='euclidean')
                

                # Optimizing k (Loop)
                p.set(message="[3/4] Optimizing k...", detail="Metrics Calculation")
                ks = list(range(2, 21))
                ch_scores, gap_scores = [], []
                
                for k in ks:
                    labels_k = fcluster(Z, k, criterion='maxclust') - 1
                    
                    if len(set(labels_k)) < 2:
                        ch_scores.append(0)
                        gap_scores.append(0)
                    else:
                        ch_scores.append(calinski_harabasz_score(X_use, labels_k))
                        
                        obs_wcss = calculate_wcss_hc(X_use, labels_k)
                        log_obs = np.log(obs_wcss + 1e-10)
                        
                        labels_ref = fcluster(Z_ref, k, criterion='maxclust') - 1
                        ref_wcss = calculate_wcss_hc(X_ref, labels_ref)
                        log_ref = np.log(ref_wcss + 1e-10)
                        
                        gap_scores.append(log_ref - log_obs)

                best_ch_k = ks[np.argmax(ch_scores)] if ch_scores else 2 #Get best CH
                best_gap_k = ks[np.argmax(gap_scores)] if gap_scores else 2 #Get best gap

                p.set(message="[4/4] Projecting...", detail="Final Viz")
                k_sel = input.hc_k()
                labels_final = fcluster(Z, k_sel, criterion='maxclust') - 1
                
                viz_meth = input.hc_viz_method()
                dims = int(input.hc_proj_dims())
                
                if viz_meth == 'pca':
                    safe_n_viz = min(dims, n_samples, X_use.shape[1])
                    emb, var_ratio_viz = compute_pca(X_use, safe_n_viz)
                    total_var_viz = np.sum(var_ratio_viz) * 100
                    cols = [f'PC{i+1}' for i in range(safe_n_viz)]
                    info_viz = f"PCA ({safe_n_viz}D): {total_var_viz:.1f}% Var."
                else:
                    try:
                        req_viz_neighbors = input.hc_viz_umap_neighbors()
                        req_viz_mindist = input.hc_viz_umap_mindist()
                    except:
                        req_viz_neighbors, req_viz_mindist = 15, 0.1

                    safe_viz_neighbors = min(req_viz_neighbors, n_samples - 2)
                    if safe_viz_neighbors < 2: safe_viz_neighbors = 2
                    
                    emb = compute_umap(X_use, n_components=dims, n_neighbors=safe_viz_neighbors, min_dist=req_viz_mindist)
                    cols = [f'UMAP{i+1}' for i in range(dims)]
                    
                    try:
                        k_neigh = min(15, n_samples - 1)
                        score = trustworthiness(X_use, emb, n_neighbors=k_neigh)
                        info_viz = f"UMAP ({dims}D): {score:.2f} (Trust)"
                    except:
                        info_viz = f"UMAP ({dims}D)"

                try:
                    idx = ks.index(k_sel)
                    curr_ch = ch_scores[idx]
                    curr_gap = gap_scores[idx]
                except:
                    pass #Avoid the crash

                hc_results.set({
                    'Z': Z, 'labels': labels_final, 'embedding': emb, 'embedding_cols': cols,
                    'sample_indices': sample_indices, 'k': k_sel, 'ks': ks,
                    'ch_scores': ch_scores, 'gap_scores': gap_scores,
                    'best_ch_k': best_ch_k, 'best_gap_k': best_gap_k,
                    'linkage': link,
                    'metrics': {
                        'info_prep': info_prep,
                        'ch': curr_ch,
                        'gap': curr_gap,
                        'info_viz': info_viz
                    }
                })
                
                final_fig = plot_cluster_map(emb, cols, labels_final, df.iloc[sample_indices])
                log_run(f"HC ({link})", info_prep, "Euclidean", k_sel, f"{curr_ch:.1f}", f"{curr_gap:.3f}", "-", info_viz, fig=final_fig)
                ui.notification_show("Hierarchical complete!", type="success")

            except Exception as e:
                print("ERRORRRRR from HC")
                traceback.print_exc()
                ui.notification_show(f"Error HC: {str(e)}", type="error")

    @render.table
    def hc_metrics_table():
        res = hc_results.get()
        if not res: return None
        
        m = res['metrics']
        df = pd.DataFrame({
            "M": ["Preprocessing", "CH Score", "Gap Stat", "Visualization"],
            "V": [
                f"{m['info_prep']}",
                f"{m['ch']:.1f}",
                f"{m['gap']:.3f}",
                m['info_viz']
            ]
        })
        df.columns = ["", ""]
        return df

    @render_widget
    def hc_metrics_plot():
        res = hc_results.get()
        if not res:
            return None
        return plot_hc_metrics(res['ks'], res['ch_scores'], res['gap_scores'], res['k'], res['best_ch_k'], res['best_gap_k'])

    @render_widget
    def hc_scatter_plot():
        res = hc_results.get()
        if not res:
            return None
        return plot_cluster_map(res['embedding'], res['embedding_cols'], res['labels'], df.iloc[res['sample_indices']])

    @render.plot
    def hc_dendrogram_plot():
        res = hc_results.get()
        if not res:
            return None
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(res['Z'], truncate_mode='lastp', p=30, show_contracted=True, leaf_rotation=90, ax=ax)
        return fig
    
    @render_widget
    def hc_univariate_plot():
        res = hc_results.get()
        return plot_cluster_univariate(df.iloc[res['sample_indices']], res['labels'], input.hc_uni_var(), input.hc_uni_subgroup(), format_label) if res else None

    # ========================================================================
    # DBSCAN LOGIC
    # ========================================================================
    
    db_results = reactive.Value(None)

    @render.ui
    def db_tabs_dynamic():
        res = db_results.get()
        current_mode = active_mode.get()
        if res is None:
            placeholder = ui.div(
                ui.markdown("Choose your parameters and click the **Run** button in the sidebar to see results"),
                style="text-align: center; min-height: 500px; display: flex; align-items: center; justify-content: center;"
            )
            content_map = placeholder
            content_uni = placeholder
        else:
            content_map = ui.layout_columns(
                ui.card(ui.card_header("Cluster Visualization"), output_widget("db_scatter_plot"), full_screen=True),
                ui.layout_column_wrap(
                    ui.card(ui.card_header("Metrics"), ui.output_table("db_metrics_table")),
                    ui.card(
                        ui.card_header("Structure (K-Distance Graph)"),
                        output_widget("db_k_dist_plot"),
                        ui.card_footer(ui.markdown("**Tip:** Knee = Epsilon"), class_="text-muted", style="font-size:0.8rem"),
                        full_screen=True
                    ),
                    width=1, heights_equal="width"
                ),
                col_widths=(8, 4)
            )
           
            content_uni = ui.card(
                ui.layout_column_wrap(
                    ui.input_selectize("db_uni_var", "Main Variable:", choices=get_clinical_choices(), selected="age_at_diagnosis"),
                    ui.input_selectize("db_uni_subgroup", "Sub-group by:", choices=get_groupby_choices(), selected="none"),
                    width=1/2
                ),
                output_widget("db_univariate_plot"),
                full_screen=True
            )

        tabs = [
            ui.nav_panel("Cluster Map", content_map),
            ui.nav_panel("Univariate", content_uni)
        ]

        if current_mode == "learning":
            guide_tab = ui.nav_panel(
                "🎓 Learner Guide",
                ui.markdown(r"""
                DBSCAN (Density-Based Spatial Clustering of Applications with Noise), unlike K-Means or Hierarchical clustering, does not force every patient into a cluster.

                It groups points that are closely packed together (high density) and marks points that lie alone in low-density regions as **outliers (Noise)**.

                <br>

                **DBSCAN relies on two main parameters:**
                * **Epsilon:** The radius around a point that defines its "neighborhood". It acts as a threshold to determine if another point is close enough to be connected.
                * **Min Samples:** The minimum number of neighbors required within that radius to form a "dense" region (a core point).

                <br>

                DBSCAN is also sensitive to the "Curse of Dimensionality". Therefore, we strongly recommend applying a Dimensionality Reduction technique before running DBSCAN.

                With DBSCAN, traditional metrics like CH and Gap Statistic are not suitable. The preferred metric is **DBCV (Density-Based Clustering Validation)**.
                * It is specifically designed for density-based clustering.
                * It ranges from -1 to 1; a positive value indicates good density separation.
                * Unlike other metrics, it works **perfectly for complex, irregular shapes**.

                <br>

                **Finding the right radius (Epsilon) is tricky.**
                * If Epsilon is too **small**, valid data is treated as Noise.
                * If Epsilon is too **large**, distinct groups merge into one giant cluster.

                To address this, users can utilize the **"Knee" Method** (k-distance graph). This involves plotting the distance to the nearest neighbors for every point, sorted from smallest to largest.
                The "knee" (point of maximum curvature) usually indicates the optimal Epsilon:
                * The flat area to the **left** of the knee represents points within clusters.
                * The area to the **right** (where the curve shoots up) represents outliers.
                """)
            )
            tabs.insert(0, guide_tab)
        return ui.navset_card_pill(*tabs, id="db_tabs")

    @reactive.Effect
    @reactive.event(input.run_dbscan)
    def compute_dbscan():
        with ui.Progress() as p:
            from sklearn.manifold import trustworthiness
            p.set(message="Processing...")
            
            X_raw, sample_indices, _ = prepare_feature_matrix(df, GENE_EXPRESSION_VARIABLES, input.db_sample_fraction()/100)
            
            prep = input.db_prep_method()
            info = "Raw Data"

            if prep == "pca":
                X_prep, var_ratio = compute_pca(X_raw, input.db_prep_pca_n())
                total_var = np.sum(var_ratio) * 100
                info = f"PCA ({input.db_prep_pca_n()}D): {total_var:.1f}% Var."
                
            elif prep == "umap":
                n_keep = input.db_prep_umap_n()
                X_prep = compute_umap(X_raw, n_keep, input.db_prep_umap_neighbors(), input.db_prep_umap_mindist())
                
                try:
                    k_neigh = min(15, len(X_prep) - 1)
                    if k_neigh > 0:
                        score = trustworthiness(X_raw, X_prep, n_neighbors=k_neigh)
                        info = f"UMAP ({n_keep}D): {score:.2f} (Trust)"
                    else:
                        info = f"UMAP ({n_keep}D): Non-linear"
                except:
                    info = f"UMAP ({n_keep}D): Non-linear"
            else:
                X_prep = X_raw

            model = DBSCAN(eps=input.db_eps(), min_samples=input.db_min_samples())
            labels = model.fit_predict(X_prep)
            
            n_clust = len(set(labels)) - (1 if -1 in labels else 0)
            noise = (np.sum(labels == -1) / len(labels)) * 100
            
            # Visualization
            viz = input.db_viz_method()
            dim = int(input.db_proj_dims())
            v_inf = ""

            if viz == 'pca':
                emb, var_ratio_viz = compute_pca(X_prep, n_components=dim)
                total_var_viz = np.sum(var_ratio_viz) * 100
                cols = [f'PC{i+1}' for i in range(dim)]
                v_inf = f"PCA ({dim}D): {total_var_viz:.1f}% Var."
            else:
                emb = compute_umap(X_prep, dim, input.db_viz_umap_neighbors(), input.db_viz_umap_min_dist())
                cols = [f'UMAP{i+1}' for i in range(dim)]
                
                # Metric UMAP Viz
                try:
                    k_neigh = min(15, len(X_prep) - 1)
                    if k_neigh > 0:
                        score = trustworthiness(X_prep, emb, n_neighbors=k_neigh)
                        v_inf = f"UMAP ({dim}D): {score:.2f} (Trust)"
                    else:
                        v_inf = f"UMAP ({dim}D): Non-linear"
                except:
                    v_inf = f"UMAP ({dim}D): Non-linear"

            # Metrics (DBCV)
            unique_labels = set(labels)
            if -1 in unique_labels: unique_labels.remove(-1)
            
            if len(unique_labels) >= 2:
                db_idx = davies_bouldin_score(X_prep, labels)
                try:
                    dbcv_score = calculate_dbcv_manual(X_prep, labels)
                except Exception as e:
                    print(f"Erreur DBCV: {e}")
                    dbcv_score = np.nan
            else:
                dbcv_score, db_idx = np.nan, np.nan

            db_results.set({
                'X_prep': X_prep, 'labels': labels, 'embedding': emb, 'embedding_cols': cols,
                'sample_indices': sample_indices, 'min_samples': input.db_min_samples(),
                'metrics': {
                    'n': n_clust, 'noise': noise, 'dbcv': dbcv_score, 'db': db_idx, 
                    'info': info, 'info_viz': v_inf
                }
            })
            
            dbcv_str = f"{dbcv_score:.3f}" if not np.isnan(dbcv_score) else "N/A"
            final_fig = plot_cluster_map(emb, cols, labels, df.iloc[sample_indices])
            log_run("DBSCAN", info, f"Eps={input.db_eps()}", n_clust, "-", "-", f"DBCV:{dbcv_str}", v_inf, fig=final_fig)
            ui.notification_show(f"DBSCAN: {n_clust} clusters", type="success")
   

    @reactive.Effect
    def update_dbscan_epsilon_range(): #We make the choice of epsilon reactive to the number of dimensions choosen in the preprocessing step
        method = input.db_prep_method()
        new_min, new_max, new_val = 0.1, 10.0, 1.0
       
        if method == "none":
            new_min, new_max, new_val = 15.0, 35.0, 25.0
        else:
            if method == "pca":
                raw_dims = input.db_prep_pca_n()
            else:
                raw_dims = input.db_prep_umap_n()

            dims = raw_dims if raw_dims is not None else 0
           
            if dims >= 50:
                new_min, new_max, new_val = 5.0, 20.0, 8.0
            elif dims >= 30:
                new_min, new_max, new_val = 2.5, 15.0, 5.0
            elif dims >= 20:
                new_min, new_max, new_val = 1.5, 12.0, 3.0
            elif dims >= 10:
                new_min, new_max, new_val = 0.1, 10.0, 1.0
            else:
                new_min, new_max, new_val = 0.1, 8.0, 0.5
        ui.update_slider("db_eps", min=new_min, max=new_max, value=new_val, step=0.1)

    @render.table
    def db_metrics_table():
        res = db_results.get()
        if not res: return None
        
        m = res['metrics']
        df = pd.DataFrame({
            "M": ["Preprocessing", "Noise Ratio", "DBCV Score", "Visualization"],
            "V": [
                m['info'], 
                f"{m['noise']:.1f}%", 
                f"{m['dbcv']:.3f}" if not np.isnan(m['dbcv']) else "N/A", 
                m['info_viz'] 
            ]
        })
        df.columns = ["", ""]
        return df
    

    @render_widget
    def db_k_dist_plot():
        res = db_results.get()
        return plot_k_distance(res['X_prep'], res['min_samples']) if res else None

    @render_widget
    def db_scatter_plot():
        res = db_results.get()
        return plot_cluster_map(res['embedding'], res['embedding_cols'], res['labels'], df.iloc[res['sample_indices']]) if res else None

    @render_widget
    def db_univariate_plot():
        res = db_results.get()
        return plot_cluster_univariate(df.iloc[res['sample_indices']], res['labels'], input.db_uni_var(), input.db_uni_subgroup(), format_label) if res else None

    @render.ui
    def comparison_plots_layout():
        plots_data = recent_plots.get()
       
        if not plots_data:
            return ui.div(
                ui.markdown("No runs yet. Run any clustering analysis (K-Means, HC, or DBSCAN) to see comparisons here."),
                style="text-align: center; color: #6c757d; padding: 50px;"
            )
           
        cards = []
        for p in plots_data:
            pass

        return ui.layout_column_wrap(
            ui.card(ui.card_header("Run 1 (Newest)"), output_widget("comp_plot_1")),
            ui.card(ui.card_header("Run 2"), output_widget("comp_plot_2")),
            ui.card(ui.card_header("Run 3 (Oldest)"), output_widget("comp_plot_3")),
            width=1/3
        )

    @render_widget
    def comp_plot_1():
        plots = recent_plots.get()
        if len(plots) > 0 and plots[0]['fig'] is not None:
            return plots[0]['fig']
        return None

    @render_widget
    def comp_plot_2():
        plots = recent_plots.get()
        if len(plots) > 1 and plots[1]['fig'] is not None:
            return plots[1]['fig']
        return None
        
    @render_widget
    def comp_plot_3():
        plots = recent_plots.get()
        if len(plots) > 2 and plots[2]['fig'] is not None:
            return plots[2]['fig']
        return None

# ============================================================================
# APP
# ============================================================================

app = App(app_ui, server)