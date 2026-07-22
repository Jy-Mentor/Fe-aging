###############################################################################
# Advanced_Figures_Generator.R
# Publication-grade (Nature/Science-level) multi-omics figures for the
# Ferro-aging project. Reads ONLY real data files; no simulation.
# Output: D:/铁衰老 绝不重蹈覆辙/advanced/Figure{1..5}_*.pdf + .png
###############################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(ComplexHeatmap)
  library(circlize)
  library(ggExtra)
  library(ggridges)
  library(svglite)
  library(cowplot)
  library(magick)
  library(scales)
  library(viridis)
  library(RColorBrewer)
  library(ggpubr)
  library(ggrepel)
  library(ggsci)
  library(grid)
})

## ---------------------------------------------------------------------------
## Paths & output directory
## ---------------------------------------------------------------------------
base_dir <- "D:/铁衰老 绝不重蹈覆辙/L2"
out_dir  <- "D:/铁衰老 绝不重蹈覆辙/advanced"
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}
if (!dir.exists(out_dir)) stop("Failed to create output directory: ", out_dir)

lfc_matrix_path  <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/02_bulk_ferroaging_lfc_matrix.csv")
deg_path         <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/02_bulk_all_degs.csv")
spatial_rds_path <- file.path(base_dir, "multi_omics_pipeline/outputs/rds/10_spatial_with_proportions.rds")
spatial_scores_path <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/06_spatial_region_scores.csv")
neuron_fp_path   <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/10_neuron_prop_vs_ferroptosis.csv")
sc_rds_path      <- file.path(base_dir, "multi_omics_pipeline/outputs/rds/08_sc_seurat_annotated_scored.rds")
augur_path       <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/09_augur_auc_ranking.csv")
pseudotime_path  <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/09_pseudotime_neuron_scores.csv")
metab_long_path  <- file.path(base_dir, "multi_omics_pipeline/data/metabolomics/ST001637_abundance_long.csv")
metab_meta_path  <- file.path(base_dir, "multi_omics_pipeline/data/metabolomics/ST001637_sample_meta.csv")
gsea_path        <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/03_bulk_gsea_all_terms.csv")
shared_pw_path   <- file.path(base_dir, "multi_omics_pipeline/output/kegg_pathway_integration/tables/cross_omics_shared_pathways.csv")
axis_match_path  <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/13_pathway_axis_match_rate.csv")
axis_table_path  <- file.path(base_dir, "multi_omics_pipeline/output/cross_omics_integration/tables/cross_omics_axis_table.csv")

for (p in c(lfc_matrix_path, deg_path, spatial_rds_path, spatial_scores_path,
            neuron_fp_path, sc_rds_path, augur_path, pseudotime_path,
            metab_long_path, metab_meta_path, gsea_path, shared_pw_path,
            axis_match_path, axis_table_path)) {
  if (!file.exists(p)) stop("Required data file missing: ", p)
}

## ---------------------------------------------------------------------------
## Logging helpers
## ---------------------------------------------------------------------------
log_info <- function(...) message("[INFO] ", ..., " [", format(Sys.time(), "%H:%M:%S"), "]")
log_warn <- function(...) message("[WARN] ", ..., " [", format(Sys.time(), "%H:%M:%S"), "]")

## ---------------------------------------------------------------------------
## Device / font setup
## ---------------------------------------------------------------------------
use_cairo <- isTRUE(capabilities("cairo"))
pdf_device <- if (use_cairo) cairo_pdf else pdf
log_info("cairo capability: ", use_cairo, " -> PDF device: ",
         if (use_cairo) "cairo_pdf" else "pdf")

font_family <- "Arial"
af <- tryCatch(grDevices::checkInvFonts("Arial") , error = function(e) NULL)
if (!("Arial" %in% names(windowsFonts()))) {
  log_warn("Arial not registered in windowsFonts(); using 'sans' fallback.")
  font_family <- "sans"
}
log_info("Font family: ", font_family)

## ---------------------------------------------------------------------------
## Seurat accessors (S4 slot based, no Seurat package dependency)
## ---------------------------------------------------------------------------
so_cells <- function(obj) rownames(obj@meta.data)
so_meta_col <- function(obj, col) obj@meta.data[[col]]
so_embedding <- function(obj, reduction = "umap") {
  red <- obj@reductions[[reduction]]
  as.matrix(red@cell.embeddings)
}
so_assay_data <- function(obj, assay = NULL, layer = "data") {
  if (is.null(assay)) assay <- obj@active.assay
  asy <- obj@assays[[assay]]
  if (.hasSlot(asy, "layers") && length(asy@layers) > 0) {
    ln <- names(asy@layers)
    target <- paste0(assay, ".", layer)
    chosen <- if (target %in% ln) target else if (layer %in% ln) layer else ln[grep(layer, ln, ignore.case = TRUE)[1]]
    mat <- asy@layers[[chosen]]
    cells <- rownames(obj@meta.data)
    cn <- colnames(mat)
    if (!is.null(cn) && length(intersect(cn, cells)) > length(cells) / 2) mat <- t(mat)
    return(as(mat, "dgCMatrix"))
  }
  if (.hasSlot(asy, layer)) return(slot(asy, layer))
  stop("assay layer not found")
}
so_fetch_gene <- function(obj, gene, assay = NULL, layer = "data") {
  mat <- so_assay_data(obj, assay = assay, layer = layer)
  if (!(gene %in% rownames(mat))) stop("Gene not found in assay: ", gene)
  as.numeric(mat[gene, ])
}
so_spatial_coords_all <- function(obj) {
  do.call(rbind, lapply(names(obj@images), function(im) {
    coords <- obj@images[[im]]@coordinates
    data.frame(spot_id = rownames(coords), imagerow = coords$imagerow,
               imagecol = coords$imagecol, tissue = coords$tissue,
               sample = im, stringsAsFactors = FALSE)
  }))
}

## ---------------------------------------------------------------------------
## Theme & palettes
## ---------------------------------------------------------------------------
theme_advanced <- function(base_size = 10) {
  theme_classic(base_size = base_size, base_family = font_family) +
    theme(
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.text = element_text(color = "black", size = rel(0.95)),
      axis.title = element_text(color = "black", size = rel(1.15), face = "bold"),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      axis.ticks.length = unit(2, "pt"),
      plot.tag = element_text(size = rel(1.8), face = "bold", family = font_family),
      plot.title = element_text(size = rel(1.3), face = "bold", hjust = 0),
      plot.subtitle = element_text(size = rel(0.9), color = "grey40", hjust = 0),
      legend.text = element_text(color = "black", size = rel(0.8)),
      legend.title = element_text(color = "black", size = rel(0.9), face = "bold"),
      legend.key.size = unit(0.4, "cm"),
      legend.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid = element_blank(),
      strip.background = element_rect(fill = "grey92", color = NA),
      strip.text = element_text(size = rel(1.0), face = "bold", color = "black"),
      plot.margin = margin(6, 8, 6, 6, "mm")
    )
}

pal_nature <- c("#4DBBD5", "#E64B35", "#00A087", "#3C5488",
                "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
                "#7E6148", "#B09C85")  # 10-color NPG-extended (CVD-safe)
pal_div    <- c("#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B")
pal_seq    <- c("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
pal_region <- c("Healthy" = "#2171B5", "Penumbra" = "#B2182B", "Other" = "#BDBDBD")
pal_condition <- c("Ctrl" = "#4DBBD5", "1DPI" = "#E64B35", "3DPI" = "#F39B7F", "7DPI" = "#3C5488")
pal_signatures <- c("Ferroptosis" = "#E64B35", "Senescence" = "#3C5488",
                    "Ferrosenescence" = "#7B3294", "Ferroaging" = "#00A087")
pal_timepoint <- c("3h" = "#4DBBD5", "12h" = "#91D1C2", "24h" = "#F39B7F",
                   "3D" = "#E64B35", "7D" = "#3C5488")
pal_evidence <- c("Strong" = "#E64B35", "Moderate" = "#F39B7F", "Weak" = "#8491B4")

sig_stars <- function(p) {
  ifelse(is.na(p), "",
  ifelse(p < 0.001, "***",
  ifelse(p < 0.01,  "**",
  ifelse(p < 0.05,  "*", ""))))
}

## ---------------------------------------------------------------------------
## Save helpers
## ---------------------------------------------------------------------------
save_plot <- function(plot, name, width, height) {
  pdf_path <- file.path(out_dir, paste0(name, ".pdf"))
  png_path <- file.path(out_dir, paste0(name, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = pdf_device, bg = "white")
  ggsave(png_path, plot, width = width, height = height, dpi = 300, bg = "white")
  log_info("saved ", name, ".pdf/.png (", width, "x", height, "in)")
}

save_heatmap <- function(ht, name, width, height) {
  pdf_path <- file.path(out_dir, paste0(name, ".pdf"))
  png_path <- file.path(out_dir, paste0(name, ".png"))
  pdf_device(pdf_path, width = width, height = height)
  draw(ht)
  dev.off()
  png(png_path, width = width, height = height, units = "in", res = 300, bg = "white")
  draw(ht)
  dev.off()
  log_info("saved heatmap ", name, ".pdf/.png (", width, "x", height, "in)")
}

tag_theme <- theme(plot.tag = element_text(size = rel(1.8), face = "bold", family = font_family))

###############################################################################
## DATA LOADING
###############################################################################
log_info("Loading data files...")

lfc_matrix <- read.csv(lfc_matrix_path, stringsAsFactors = FALSE, row.names = 1)
tp_cols <- intersect(c("X3h","X12h","X24h","X3D","X7D"), colnames(lfc_matrix))
lfc_matrix <- lfc_matrix[, tp_cols, drop = FALSE]
lfc_matrix <- lfc_matrix[apply(is.finite(as.matrix(lfc_matrix)), 1, all), , drop = FALSE]
log_info("LFC matrix: ", nrow(lfc_matrix), " genes x ", ncol(lfc_matrix), " timepoints")

spatial_obj <- readRDS(spatial_rds_path)
spatial_scores <- read.csv(spatial_scores_path, stringsAsFactors = FALSE)
rownames(spatial_scores) <- spatial_scores$spot_id

neuron_fp <- read.csv(neuron_fp_path, stringsAsFactors = FALSE)

sc_obj <- readRDS(sc_rds_path)

augur <- read.csv(augur_path, stringsAsFactors = FALSE)

pseudotime_raw <- read.csv(pseudotime_path, stringsAsFactors = FALSE)
n_inf <- sum(is.infinite(pseudotime_raw$pseudotime))
if (n_inf > 0) log_warn("Filtering ", n_inf, " Inf pseudotime rows out of ", nrow(pseudotime_raw),
                        " (these are un-rooted cells with no trajectory assignment)")
pseudotime <- pseudotime_raw[is.finite(pseudotime_raw$pseudotime), , drop = FALSE]
if (nrow(pseudotime) == 0) stop("No finite pseudotime values remain after filtering Inf")
log_info("Pseudotime: ", nrow(pseudotime), " finite cells (dropped ", n_inf, " Inf)")

metab_long <- read.csv(metab_long_path, stringsAsFactors = FALSE)
metab_meta_raw <- read.csv(metab_meta_path, stringsAsFactors = FALSE)

gsea <- read.csv(gsea_path, stringsAsFactors = FALSE)
shared_pw <- read.csv(shared_pw_path, stringsAsFactors = FALSE)
axis_match <- read.csv(axis_match_path, stringsAsFactors = FALSE)
axis_table <- read.csv(axis_table_path, stringsAsFactors = FALSE)

## ---------------------------------------------------------------------------
## Spatial plot data (coords + scores + neuron proportion)
## ---------------------------------------------------------------------------
spatial_cells <- rownames(spatial_obj@meta.data)
common_spots <- intersect(spatial_cells, spatial_scores$spot_id)
if (length(common_spots) == 0) stop("No common spots between spatial Seurat and region scores")
log_info("Spatial common spots: ", length(common_spots), " / ", length(spatial_cells))

spatial_meta <- spatial_obj@meta.data[common_spots, , drop = FALSE]
spatial_meta$spot_id <- common_spots
spatial_meta$region <- spatial_scores[common_spots, "region"]
spatial_meta$Ferroaging <- spatial_scores[common_spots, "Ferroaging"]
spatial_meta$Ferroptosis <- spatial_scores[common_spots, "Ferroptosis"]
neuron_cols <- intersect(c("prop_NeuronsGABA","prop_NeuronsGLUT"), colnames(spatial_meta))
if (length(neuron_cols) == 2) {
  spatial_meta$neuron_prop <- spatial_meta[[neuron_cols[1]]] + spatial_meta[[neuron_cols[2]]]
} else {
  stop("Neuron proportion columns (prop_NeuronsGABA/prop_NeuronsGLUT) not found in spatial meta.data")
}

spatial_coords_all <- so_spatial_coords_all(spatial_obj)
spatial_coords_all <- spatial_coords_all[spatial_coords_all$spot_id %in% common_spots, , drop = FALSE]
spatial_plot_df <- merge(spatial_coords_all,
                         spatial_meta[, c("spot_id","region","Ferroptosis","Ferroaging","neuron_prop")],
                         by = "spot_id", all.x = TRUE)
spatial_plot_df <- spatial_plot_df[!is.na(spatial_plot_df$Ferroaging), , drop = FALSE]
spatial_plot_df$sample <- factor(spatial_plot_df$sample)
log_info("Spatial plot df: ", nrow(spatial_plot_df), " spots across ",
         nlevels(spatial_plot_df$sample), " samples")

## ---------------------------------------------------------------------------
## Single-cell UMAP + scores
## ---------------------------------------------------------------------------
sc_umap <- so_embedding(sc_obj, "umap")
colnames(sc_umap) <- c("UMAP1","UMAP2")
sc_meta <- sc_obj@meta.data
sc_common <- intersect(rownames(sc_meta), rownames(sc_umap))
sc_umap <- sc_umap[sc_common, , drop = FALSE]
sc_meta <- sc_meta[sc_common, , drop = FALSE]
sc_df <- data.frame(
  UMAP1 = sc_umap[, 1], UMAP2 = sc_umap[, 2],
  Celltypes = sc_meta$Celltypes,
  Condition = sc_meta$Condition,
  Ferroptosis_UCell = sc_meta$Ferroptosis_UCell,
  Senescence_UCell = sc_meta$Senescence_UCell,
  Ferroaging_UCell = sc_meta$Ferroaging_UCell,
  Ferrosenescence_UCell = sc_meta$Ferrosenescence_UCell,
  stringsAsFactors = FALSE
)
sc_df <- sc_df[complete.cases(sc_df[, c("UMAP1","UMAP2","Ferroaging_UCell")]), , drop = FALSE]
log_info("SC UMAP df: ", nrow(sc_df), " cells; ", nlevels(factor(sc_df$Celltypes)), " cell types")

## ---------------------------------------------------------------------------
## Metabolomics differential (Young=3wk vs Old=59wk)
## ---------------------------------------------------------------------------
metab_meta <- metab_meta_raw %>%
  dplyr::filter(!is.na(Age), Age %in% c("3 weeks","59 weeks")) %>%
  dplyr::select(sample_id, Age) %>% dplyr::distinct() %>%
  dplyr::mutate(age_group = ifelse(Age == "3 weeks", "Young", "Old"))

metab_long_f <- metab_long %>%
  dplyr::inner_join(metab_meta %>% dplyr::select(sample_id, age_group), by = "sample_id") %>%
  dplyr::filter(abundance > 0)

metab_res <- metab_long_f %>%
  dplyr::group_by(metabolite) %>%
  dplyr::filter(dplyr::n_distinct(age_group) == 2,
                sum(age_group == "Young") >= 2,
                sum(age_group == "Old") >= 2,
                dplyr::n() >= 6) %>%
  dplyr::summarise(
    mean_young = mean(abundance[age_group == "Young"], na.rm = TRUE),
    mean_old   = mean(abundance[age_group == "Old"], na.rm = TRUE),
    sem_log2FC = sqrt((sd(abundance[age_group == "Old"], na.rm = TRUE) /
                         (mean(abundance[age_group == "Old"], na.rm = TRUE) * log(2) *
                          sqrt(sum(age_group == "Old"))))^2 +
                        (sd(abundance[age_group == "Young"], na.rm = TRUE) /
                         (mean(abundance[age_group == "Young"], na.rm = TRUE) * log(2) *
                          sqrt(sum(age_group == "Young"))))^2),
    log2FC = log2(mean_old / mean_young),
    pvalue = t.test(abundance ~ age_group)$p.value,
    .groups = "drop") %>%
  dplyr::mutate(padj = p.adjust(pvalue, method = "BH"),
                sig = sig_stars(padj))
metab_res <- metab_res[is.finite(metab_res$log2FC), , drop = FALSE]
log_info("Metabolomics differential: ", nrow(metab_res), " metabolites tested")

###############################################################################
## FIGURE 1: Multi-omics overview (2x2, 7.2 x 7 in)
###############################################################################
log_info("Building Figure 1 (multi-omics overview)...")

## 1A: Spatial Ferroaging scatter (facet by sample)
f1a <- ggplot(spatial_plot_df, aes(x = imagecol, y = imagerow, color = Ferroaging)) +
  geom_point(size = 1.8, alpha = 0.85) +
  scale_y_reverse() +
  scale_color_gradientn(colors = pal_seq, name = "Ferroaging") +
  facet_wrap(~ sample, nrow = 1) +
  coord_fixed(ratio = 1) +
  labs(x = NULL, y = NULL, title = "Spatial Ferroaging score") +
  theme_advanced() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        axis.line = element_blank(),
        strip.text = element_text(size = rel(0.7)))

## 1B: SC UMAP Ferroaging
f1b <- ggplot(sc_df, aes(x = UMAP1, y = UMAP2, color = Ferroaging_UCell)) +
  geom_point(size = 1.2, alpha = 0.8) +
  scale_color_gradientn(colors = pal_seq, name = "Ferroaging") +
  labs(x = "UMAP 1", y = "UMAP 2", title = "Single-cell Ferroaging (UCell)") +
  theme_advanced()

## 1C: Top12 metabolite log2FC waterfall
metab_top <- metab_res %>%
  dplyr::arrange(padj, dplyr::desc(abs(log2FC))) %>%
  head(12) %>%
  dplyr::arrange(log2FC) %>%
  dplyr::mutate(metabolite = factor(metabolite, levels = metabolite))
metab_lims <- max(abs(metab_top$log2FC), na.rm = TRUE) * 1.3
f1c <- ggplot(metab_top, aes(x = metabolite, y = log2FC, fill = log2FC)) +
  geom_col(width = 0.7, color = "black", linewidth = 0.2) +
  geom_errorbar(aes(ymin = log2FC - sem_log2FC, ymax = log2FC + sem_log2FC),
                width = 0.25, color = "black", linewidth = 0.3) +
  geom_text(aes(y = log2FC + ifelse(log2FC >= 0, sem_log2FC, -sem_log2FC),
                label = sig),
            vjust = ifelse(metab_top$log2FC >= 0, -0.4, 1.4), size = 3) +
  scale_fill_gradientn(colors = pal_div, limits = c(-metab_lims, metab_lims), guide = "none") +
  coord_flip() +
  labs(x = NULL, y = "log2FC (Old / Young)", title = "Top 12 altered metabolites") +
  theme_advanced() +
  theme(axis.text.y = element_text(size = rel(0.7)))

## 1D: GSEA NES heatmap (4 signatures x comparisons)
gsea_focus <- gsea %>%
  dplyr::filter(Description %in% names(pal_signatures)) %>%
  dplyr::mutate(Description = factor(Description, levels = names(pal_signatures)),
                comparison = factor(comparison, levels = c("3h","12h","3D")),
                sig = sig_stars(p.adjust),
                NES_lab = sprintf("%.2f", NES))
gsea_lims <- max(abs(gsea_focus$NES), na.rm = TRUE)
f1d <- ggplot(gsea_focus, aes(x = comparison, y = Description, fill = NES)) +
  geom_tile(color = "white", linewidth = 1.1) +
  geom_text(aes(label = NES_lab), size = 2.8, color = "black", fontface = "bold") +
  geom_text(aes(y = as.numeric(Description) + 0.32, label = sig), size = 3.2, color = "black") +
  scale_fill_gradientn(colors = pal_div, limits = c(-gsea_lims, gsea_lims), name = "NES") +
  labs(x = "Timepoint", y = NULL, title = "GSEA NES (FA signatures)") +
  theme_advanced() +
  theme(axis.text.x = element_text(size = rel(0.85)),
        panel.grid = element_blank())

fig1 <- (f1a + f1b) / (f1c + f1d) +
  plot_layout(heights = c(1, 1)) +
  plot_annotation(tag_levels = "A", tag_suffix = ")") &
  tag_theme
save_plot(fig1, "Figure1_multimics_integration", width = 7.2, height = 7)

###############################################################################
## FIGURE 2: Bulk time-series GSEA + heatmap (7.2 x 4 in)
###############################################################################
log_info("Building Figure 2 (bulk time-series)...")

## 2A: FA gene log2FC ridge density (5 timepoints)
lfc_long <- lfc_matrix %>%
  tibble::rownames_to_column("gene") %>%
  tidyr::pivot_longer(cols = dplyr::all_of(tp_cols), names_to = "timepoint",
                      values_to = "log2FC") %>%
  dplyr::mutate(timepoint = factor(gsub("^X","",timepoint),
                                   levels = c("3h","12h","24h","3D","7D")))
f2a <- ggplot(lfc_long, aes(x = log2FC, y = timepoint, fill = after_stat(x))) +
  geom_density_ridges_gradient(bandwidth = 0.005, rel_min_height = 0.01,
                               scale = 2.2, color = "black", linewidth = 0.25) +
  scale_fill_gradientn(colors = pal_div, name = "log2FC") +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = "log2 Fold Change (FA genes)", y = "Timepoint",
       title = "FA gene log2FC distribution") +
  theme_advanced()

## 2B: FA gene log2FC heatmap (ComplexHeatmap, kmeans=3)
mat <- as.matrix(lfc_matrix)
colnames(mat) <- gsub("^X","",colnames(mat))
set.seed(42)
km_clusters <- kmeans(mat, centers = 3)$cluster
mat_lims <- max(abs(mat), na.rm = TRUE)
col_fun <- circlize::colorRamp2(c(-mat_lims, 0, mat_lims),
                                c("#2166AC","#F7F7F7","#B2182B"))
tp_levels <- colnames(mat)
col_anno <- HeatmapAnnotation(
  Timepoint = factor(tp_levels, levels = tp_levels),
  col = list(Timepoint = setNames(pal_timepoint[tp_levels], tp_levels)),
  show_annotation_name = FALSE)
row_anno <- rowAnnotation(
  Cluster = factor(km_clusters),
  col = list(Cluster = setNames(pal_nature[1:3], sort(unique(km_clusters)))),
  show_annotation_name = FALSE)
ht2 <- Heatmap(mat, name = "log2FC", col = col_fun,
  row_split = factor(km_clusters, levels = sort(unique(km_clusters))),
  row_title = "Cluster %s", row_title_gp = gpar(fontsize = 8, fontface = "bold"),
  cluster_columns = FALSE, cluster_row_slices = TRUE,
  show_row_names = FALSE, show_column_names = TRUE,
  column_names_gp = gpar(fontsize = 8),
  top_annotation = col_anno, left_annotation = row_anno,
  column_title = "FA gene log2FC across timepoints",
  column_title_gp = gpar(fontsize = 10, fontface = "bold"),
  border = TRUE, use_raster = TRUE, raster_quality = 4)

ht2_grob <- grid::grid.grabExpr(draw(ht2))
f2b <- wrap_elements(full = ht2_grob)

fig2 <- (f2a + plot_spacer()) / f2b +
  plot_layout(heights = c(1, 1.6), widths = c(1, 1e-4)) +
  plot_annotation(tag_levels = "A", tag_suffix = ")") &
  tag_theme
save_plot(fig2, "Figure2_bulk_gsea", width = 7.2, height = 4)
save_heatmap(ht2, "Figure2B_bulk_heatmap_standalone", width = 5, height = 4.5)

###############################################################################
## FIGURE 3: Spatial transcriptomics (2x2, 7.2 x 7 in)
###############################################################################
log_info("Building Figure 3 (spatial transcriptomics)...")

## 3A: Spatial Ferroptosis scatter (facet by sample)
f3a <- ggplot(spatial_plot_df, aes(x = imagecol, y = imagerow, color = Ferroptosis)) +
  geom_point(size = 1.8, alpha = 0.85) +
  scale_y_reverse() +
  scale_color_gradientn(colors = pal_seq, name = "Ferroptosis") +
  facet_wrap(~ sample, nrow = 1) +
  coord_fixed(ratio = 1) +
  labs(x = NULL, y = NULL, title = "Spatial Ferroptosis score") +
  theme_advanced() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        axis.line = element_blank(),
        strip.text = element_text(size = rel(0.7)))

## 3B: Spatial region scatter (pal_region)
f3b <- ggplot(spatial_plot_df, aes(x = imagecol, y = imagerow, color = region)) +
  geom_point(size = 1.8, alpha = 0.85) +
  scale_y_reverse() +
  scale_color_manual(values = pal_region, name = "Region") +
  facet_wrap(~ sample, nrow = 1) +
  coord_fixed(ratio = 1) +
  labs(x = NULL, y = NULL, title = "Spatial region classification") +
  theme_advanced() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        axis.line = element_blank(),
        strip.text = element_text(size = rel(0.7)),
        legend.position = "right")

## 3C: Region vs Ferroaging violin + box + jitter
region_levels <- c("Healthy","Penumbra","Other")
spatial_plot_df$region <- factor(spatial_plot_df$region, levels = region_levels)
f3c <- ggplot(spatial_plot_df, aes(x = region, y = Ferroaging, fill = region)) +
  geom_violin(alpha = 0.45, linewidth = 0.3, trim = TRUE) +
  geom_boxplot(width = 0.22, outlier.shape = NA, alpha = 0.9, color = "black", linewidth = 0.3) +
  geom_jitter(width = 0.12, size = 0.3, alpha = 0.25, color = "grey30") +
  scale_fill_manual(values = pal_region, guide = "none") +
  stat_compare_means(comparisons = list(c("Healthy","Penumbra"),
                                        c("Healthy","Other"),
                                        c("Penumbra","Other")),
                     method = "wilcox.test", size = 2.6, tip_length = 0.01) +
  labs(x = "Region", y = "Ferroaging score", title = "Ferroaging by region") +
  theme_advanced()

## 3D: Neuron proportion vs Ferroptosis correlation (marginal density)
neuron_fp <- neuron_fp[is.finite(neuron_fp$neuron_prop) & is.finite(neuron_fp$fp_score), , drop = FALSE]
f3d_base <- ggplot(neuron_fp, aes(x = neuron_prop, y = fp_score)) +
  geom_point(size = 0.8, alpha = 0.6, color = "#3C5488") +
  geom_smooth(method = "lm", se = TRUE, color = "#E64B35", fill = "#E64B35",
              linewidth = 0.6, alpha = 0.15) +
  stat_cor(method = "spearman", label.x = 0.02, label.y.npc = "top", size = 3,
           color = "black") +
  labs(x = "Neuron proportion", y = "Ferroptosis score",
       title = "Neuron loss vs ferroptosis") +
  theme_advanced()
f3d <- ggMarginal(f3d_base, type = "density", fill = "#4DBBD5", color = "#3C5488",
                  alpha = 0.5, size = 2.5)

fig3 <- (f3a + f3b) / (f3c + wrap_elements(full = f3d)) +
  plot_layout(heights = c(1, 1)) +
  plot_annotation(tag_levels = "A", tag_suffix = ")") &
  tag_theme
save_plot(fig3, "Figure3_spatial_transcriptomics", width = 7.2, height = 7)

###############################################################################
## FIGURE 4: Single cell (7.2 x 9 in)
###############################################################################
log_info("Building Figure 4 (single-cell)...")

## 4A: UMAP cell types (pal_nature)
ct_levels <- names(sort(table(sc_df$Celltypes), decreasing = TRUE))
sc_df$Celltypes <- factor(sc_df$Celltypes, levels = ct_levels)
f4a <- ggplot(sc_df, aes(x = UMAP1, y = UMAP2, color = Celltypes)) +
  geom_point(size = 1.2, alpha = 0.8) +
  scale_color_manual(values = setNames(pal_nature, ct_levels), name = "Cell type") +
  labs(x = "UMAP 1", y = "UMAP 2", title = "Cell type UMAP") +
  theme_advanced() +
  guides(color = guide_legend(override.aes = list(size = 2.5, alpha = 1),
                              ncol = 1, keywidth = 0.4, keyheight = 0.4))

## 4B: Ferroaging ridge by cell type (pal_div gradient)
f4b <- ggplot(sc_df, aes(x = Ferroaging_UCell, y = Celltypes, fill = after_stat(x))) +
  geom_density_ridges_gradient(scale = 1.6, rel_min_height = 0.01,
                               color = "black", linewidth = 0.2) +
  scale_fill_gradientn(colors = pal_div, name = "Ferroaging") +
  labs(x = "Ferroaging UCell score", y = NULL, title = "Ferroaging by cell type") +
  theme_advanced()

## 4C: Pairwise correlations (Ferroptosis / Senescence / Sat1)
sat1_expr <- so_fetch_gene(sc_obj, "Sat1")
names(sat1_expr) <- rownames(sc_obj@meta.data)
sat1_df <- data.frame(cell = names(sat1_expr), Sat1 = as.numeric(sat1_expr),
                      stringsAsFactors = FALSE)
corr_df <- sc_df %>%
  tibble::rownames_to_column("cell") %>%
  dplyr::inner_join(sat1_df, by = "cell")
f4c1 <- ggplot(corr_df, aes(x = Ferroptosis_UCell, y = Senescence_UCell)) +
  geom_point(size = 0.6, alpha = 0.5, color = "#3C5488") +
  geom_smooth(method = "lm", se = TRUE, color = "#E64B35", fill = "#E64B35",
              linewidth = 0.5, alpha = 0.12) +
  stat_cor(method = "spearman", size = 2.6, label.x.npc = 0.02, label.y.npc = "top") +
  labs(x = "Ferroptosis (UCell)", y = "Senescence (UCell)") +
  theme_advanced(base_size = 9)
f4c2 <- ggplot(corr_df, aes(x = Ferroptosis_UCell, y = Sat1)) +
  geom_point(size = 0.6, alpha = 0.5, color = "#E64B35") +
  geom_smooth(method = "lm", se = TRUE, color = "#3C5488", fill = "#3C5488",
              linewidth = 0.5, alpha = 0.12) +
  stat_cor(method = "spearman", size = 2.6, label.x.npc = 0.02, label.y.npc = "top") +
  labs(x = "Ferroptosis (UCell)", y = "Sat1 expression") +
  theme_advanced(base_size = 9)
f4c3 <- ggplot(corr_df, aes(x = Senescence_UCell, y = Sat1)) +
  geom_point(size = 0.6, alpha = 0.5, color = "#7B3294") +
  geom_smooth(method = "lm", se = TRUE, color = "#00A087", fill = "#00A087",
              linewidth = 0.5, alpha = 0.12) +
  stat_cor(method = "spearman", size = 2.6, label.x.npc = 0.02, label.y.npc = "top") +
  labs(x = "Senescence (UCell)", y = "Sat1 expression") +
  theme_advanced(base_size = 9)
f4c <- (f4c1 + f4c2 + f4c3) + plot_annotation(tag_levels = NULL) &
  theme(plot.tag = element_blank())

## 4D: Augur AUC lollipop (facet by comparison)
augur <- augur %>%
  dplyr::mutate(comparison = factor(comparison, levels = c("1DPI","3DPI","7DPI")))
f4d <- ggplot(augur, aes(x = AUC, y = reorder(cell_type, AUC), color = comparison)) +
  geom_segment(aes(x = 0.5, xend = AUC, yend = cell_type), linewidth = 0.6) +
  geom_point(size = 2.2) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey50", linewidth = 0.3) +
  scale_color_manual(values = pal_condition, name = "Condition") +
  facet_wrap(~ comparison, scales = "free_y", ncol = 1) +
  labs(x = "Augur AUC", y = NULL, title = "Cell-type priority (Augur)") +
  theme_advanced() +
  theme(strip.text = element_text(size = rel(0.8)),
        axis.text.y = element_text(size = rel(0.7)))

## 4E: Pseudotime density by Condition
pseudotime$Condition <- factor(pseudotime$Condition, levels = c("Ctrl","1DPI","3DPI","7DPI"))
f4e <- ggplot(pseudotime, aes(x = pseudotime, fill = Condition, color = Condition)) +
  geom_density(alpha = 0.4, linewidth = 0.4) +
  scale_fill_manual(values = pal_condition, name = "Condition") +
  scale_color_manual(values = pal_condition, name = "Condition") +
  labs(x = "Pseudotime", y = "Density", title = "Neuron pseudotime by condition") +
  theme_advanced()

## 4F: UCell along pseudotime LOESS (pal_signatures)
pt_long <- pseudotime %>%
  dplyr::select(pseudotime, Condition,
                Ferroptosis_UCell, Senescence_UCell,
                Ferrosenescence_UCell, Ferroaging_UCell) %>%
  tidyr::pivot_longer(cols = dplyr::ends_with("_UCell"),
                      names_to = "signature", values_to = "score") %>%
  dplyr::mutate(signature = gsub("_UCell$","", signature),
                signature = factor(signature, levels = names(pal_signatures)))
f4f <- ggplot(pt_long, aes(x = pseudotime, y = score, color = signature)) +
  geom_smooth(method = "loess", span = 0.4, se = TRUE, linewidth = 0.7,
              alpha = 0.15) +
  scale_color_manual(values = pal_signatures, name = "Signature") +
  labs(x = "Pseudotime", y = "UCell score", title = "Signature dynamics along pseudotime") +
  theme_advanced()

fig4 <- (f4a + f4b) / (f4c + f4d) / (f4e + f4f) +
  plot_layout(heights = c(1, 1, 1)) +
  plot_annotation(tag_levels = "A", tag_suffix = ")") &
  tag_theme
save_plot(fig4, "Figure4_single_cell", width = 7.2, height = 9)

###############################################################################
## FIGURE 5: Metabolomics + cross-omics (7.2 x 4.5 in)
###############################################################################
log_info("Building Figure 5 (metabolomics + cross-omics)...")

## 5A: SAT1-polyamine axis metabolite log2FC waterfall
axis_df <- axis_table %>%
  dplyr::filter(axis_name == "SAT1-polyamine") %>%
  dplyr::mutate(sig = sig_stars(p_adj_aging)) %>%
  dplyr::arrange(log2FC_aging) %>%
  dplyr::mutate(display_name = factor(display_name, levels = display_name))
ax_lims <- max(abs(axis_df$log2FC_aging), na.rm = TRUE) * 1.3
f5a <- ggplot(axis_df, aes(x = display_name, y = log2FC_aging, fill = log2FC_aging)) +
  geom_col(width = 0.7, color = "black", linewidth = 0.2) +
  geom_text(aes(y = log2FC_aging + ifelse(log2FC_aging >= 0, ax_lims*0.04, -ax_lims*0.04),
                label = sig),
            vjust = ifelse(axis_df$log2FC_aging >= 0, -0.3, 1.3), size = 3) +
  scale_fill_gradientn(colors = pal_div, limits = c(-ax_lims, ax_lims), guide = "none") +
  coord_flip() +
  labs(x = NULL, y = "log2FC (Old / Young)", title = "SAT1-polyamine axis metabolites") +
  theme_advanced() +
  theme(axis.text.y = element_text(size = rel(0.75)))

## 5B: Driver gene - KEGG pathway chord diagram (circlize)
short_pw <- function(nm) {
  s <- strsplit(nm, " - Mus musculus")[[1]][1]
  s <- strsplit(s, " \\(")[[1]][1]
  if (nchar(s) > 42) s <- paste0(substr(s, 1, 39), "...")
  s
}
drivers <- unique(axis_match$Driver_Gene)
drv_evidence <- setNames(axis_match$Evidence_Level, axis_match$Driver_Gene)
edges <- list()
for (i in seq_len(nrow(shared_pw))) {
  gl <- strsplit(shared_pw$gene_list[i], ";")[[1]]
  hits <- intersect(gl, drivers)
  if (length(hits) == 0) next
  pw_name <- short_pw(shared_pw$pathway_name[i])
  for (g in hits) {
    edges[[length(edges) + 1]] <- data.frame(from = g, to = pw_name,
                                             score = shared_pw$cross_omics_score[i],
                                             stringsAsFactors = FALSE)
  }
}
edges_df <- do.call(rbind, edges)
edges_df <- edges_df[order(-edges_df$score), , drop = FALSE]
if (nrow(edges_df) > 30) edges_df <- head(edges_df, 30)
if (nrow(edges_df) == 0) stop("No driver-gene / KEGG pathway edges found")

all_sectors <- unique(c(as.character(edges_df$from), as.character(edges_df$to)))
grid_colors <- character(length(all_sectors))
names(grid_colors) <- all_sectors
for (s in all_sectors) {
  if (s %in% drivers) {
    ev <- drv_evidence[s]
    grid_colors[s] <- pal_evidence[ifelse(is.na(ev), "Weak", ev)]
  } else {
    grid_colors[s] <- "#BDBDBD"
  }
}
chord_grob <- grid::grid.grabExpr({
  circos.clear()
  circos.par(gap.after = 2, start.degree = 90, clock.wise = FALSE)
  chordDiagram(edges_df[, c("from","to","score")], grid.col = grid_colors,
               transparency = 0.30, annotationTrack = "grid",
               preAllocateTracks = 1)
  circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    xlim <- get.cell.meta.data("xlim"); ylim <- get.cell.meta.data("ylim")
    sector.name <- get.cell.meta.data("sector.index")
    circos.text(mean(xlim), ylim[1], sector.name, facing = "clockwise",
                niceFacing = TRUE, adj = c(0, 0.5), cex = 0.55, col = "black")
  }, bg.border = NA)
  circos.clear()
})
f5b <- wrap_elements(full = chord_grob)

fig5 <- (f5a + plot_spacer()) / f5b +
  plot_layout(heights = c(1, 1), widths = c(1, 1e-4)) +
  plot_annotation(tag_levels = "A", tag_suffix = ")") &
  tag_theme
save_plot(fig5, "Figure5_metabolomics_crossomics", width = 7.2, height = 4.5)

log_info("All figures generated successfully in: ", out_dir)
