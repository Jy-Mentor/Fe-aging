###############################################################################
# Advanced_Figures_Generator.R
# Publication-grade independent figures for the Ferro-aging project.
# - NO multi-panel composition (no patchwork/cowplot assembly).
# - Each subfigure is saved as an independent PDF + PNG file.
# - All data read from REAL files; no simulation / fabrication.
# - Visual quality: theme_luxe, perceptually-uniform palettes, Arial font.
###############################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(ComplexHeatmap)
  library(circlize)
  library(ggExtra)
  library(ggridges)
  library(svglite)
  library(scales)
  library(viridis)
  library(RColorBrewer)
  library(ggsci)
  library(grid)
})

## ---------------------------------------------------------------------------
## Paths & output directory
## ---------------------------------------------------------------------------
base_dir <- "D:/铁衰老 绝不重蹈覆辙/L2"
out_dir  <- "D:/铁衰老 绝不重蹈覆辙/advanced"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(out_dir)) stop("Failed to create output directory: ", out_dir)

lfc_matrix_path    <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/02_bulk_ferroaging_lfc_matrix.csv")
spatial_rds_path   <- file.path(base_dir, "multi_omics_pipeline/outputs/rds/10_spatial_with_proportions.rds")
spatial_scores_path<- file.path(base_dir, "multi_omics_pipeline/outputs/tables/06_spatial_region_scores.csv")
neuron_fp_path     <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/10_neuron_prop_vs_ferroptosis.csv")
sc_rds_path        <- file.path(base_dir, "multi_omics_pipeline/outputs/rds/08_sc_seurat_annotated_scored.rds")
augur_path         <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/09_augur_auc_ranking.csv")
pseudotime_path    <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/09_pseudotime_neuron_scores.csv")
metab_long_path    <- file.path(base_dir, "multi_omics_pipeline/data/metabolomics/ST001637_abundance_long.csv")
metab_meta_path    <- file.path(base_dir, "multi_omics_pipeline/data/metabolomics/ST001637_sample_meta.csv")
gsea_path          <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/03_bulk_gsea_all_terms.csv")
shared_pw_path     <- file.path(base_dir, "multi_omics_pipeline/output/kegg_pathway_integration/tables/cross_omics_shared_pathways.csv")
axis_match_path    <- file.path(base_dir, "multi_omics_pipeline/outputs/tables/13_pathway_axis_match_rate.csv")
axis_table_path    <- file.path(base_dir, "multi_omics_pipeline/output/cross_omics_integration/tables/cross_omics_axis_table.csv")

required_files <- c(lfc_matrix_path, spatial_rds_path, spatial_scores_path,
                    neuron_fp_path, sc_rds_path, augur_path, pseudotime_path,
                    metab_long_path, metab_meta_path, gsea_path, shared_pw_path,
                    axis_match_path, axis_table_path)
for (p in required_files) {
  if (!file.exists(p)) stop("Required data file missing: ", p)
}

## ---------------------------------------------------------------------------
## Logging helpers
## ---------------------------------------------------------------------------
log_info <- function(...) message("[INFO] ", ..., " [", format(Sys.time(), "%H:%M:%S"), "]")
log_warn <- function(...) message("[WARN] ", ..., " [", format(Sys.time(), "%H:%M:%S"), "]")

## ---------------------------------------------------------------------------
## Font / device setup
## ---------------------------------------------------------------------------
use_cairo <- isTRUE(capabilities("cairo"))
pdf_device <- if (use_cairo) cairo_pdf else pdf
log_info("cairo capability: ", use_cairo, " -> PDF device: ",
         if (use_cairo) "cairo_pdf" else "pdf")

font_family <- "sans"
if ("Arial" %in% names(windowsFonts())) font_family <- "Arial"
log_info("Font family: ", font_family)

## ---------------------------------------------------------------------------
## Seurat S4 slot accessors (no Seurat package dependency)
## ---------------------------------------------------------------------------
so_embedding <- function(obj, reduction = "umap") {
  as.matrix(obj@reductions[[reduction]]@cell.embeddings)
}
so_assay_data <- function(obj, assay = NULL, layer = "data") {
  if (is.null(assay)) assay <- obj@active.assay
  asy <- obj@assays[[assay]]
  if (.hasSlot(asy, "layers") && length(asy@layers) > 0) {
    ln <- names(asy@layers); target <- paste0(assay, ".", layer)
    chosen <- if (target %in% ln) target else if (layer %in% ln) layer else ln[grep(layer, ln, ignore.case = TRUE)[1]]
    mat <- asy@layers[[chosen]]; cells <- rownames(obj@meta.data)
    if (!is.null(colnames(mat)) && length(intersect(colnames(mat), cells)) > length(cells) / 2) mat <- t(mat)
    return(as(mat, "dgCMatrix"))
  }
  if (.hasSlot(asy, layer)) return(slot(asy, layer))
  stop("assay layer not found")
}
so_fetch_gene <- function(obj, gene, layer = "data") {
  mat <- so_assay_data(obj, layer = layer)
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
## Theme & palettes (extremely refined)
## ---------------------------------------------------------------------------
theme_luxe <- function(base_size = 12, base_family = font_family) {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(color = "black", linewidth = 0.6),
      axis.text = element_text(color = "black", size = rel(1.0)),
      axis.title = element_text(color = "black", size = rel(1.2), face = "bold"),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.ticks.length = unit(3, "pt"),
      plot.title = element_text(size = rel(1.4), face = "bold", hjust = 0, margin = margin(b = 8)),
      plot.subtitle = element_text(size = rel(0.9), color = "grey30", hjust = 0),
      legend.text = element_text(color = "black", size = rel(0.85)),
      legend.title = element_text(color = "black", size = rel(0.95), face = "bold"),
      legend.key.size = unit(0.45, "cm"),
      legend.background = element_rect(fill = "white", color = NA),
      legend.box.background = element_rect(fill = "white", color = "grey80", linewidth = 0.3),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "grey95", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "#E8E8E8", color = NA),
      strip.text = element_text(size = rel(1.1), face = "bold", color = "black"),
      plot.margin = margin(10, 12, 10, 10, "mm")
    )
}

pal_luxe <- c("#6A9FB5", "#D4675E", "#7BAE7F", "#8B7AAE",
              "#D4A574", "#6B8E8E", "#C99DA3", "#5A8C7A",
              "#B5651D", "#999933", "#44AA99", "#882255")
pal_div_luxe <- c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#FDDBC7",
                  "#F4A582", "#D6604D", "#B2182B")
pal_region_luxe <- c("Healthy" = "#4393C3", "Penumbra" = "#B2182B", "Other" = "#D9D9D9")
pal_cond_luxe <- c("Ctrl" = "#6A9FB5", "1DPI" = "#D4675E", "3DPI" = "#D4A574", "7DPI" = "#8B7AAE")
pal_sig_luxe <- c("Ferroptosis" = "#D4675E", "Senescence" = "#6A9FB5",
                  "Ferrosenescence" = "#8B7AAE", "Ferroaging" = "#7BAE7F")
pal_evidence_luxe <- c("Strong" = "#B2182B", "Moderate" = "#D4A574", "Weak" = "#8B7AAE")

sig_stars <- function(p) {
  ifelse(is.na(p), "",
  ifelse(p < 0.001, "***",
  ifelse(p < 0.01,  "**",
  ifelse(p < 0.05,  "*", ""))))
}

## ---------------------------------------------------------------------------
## Save helpers
## ---------------------------------------------------------------------------
save_single <- function(plot, filename, width, height) {
  pdf_path <- file.path(out_dir, paste0(filename, ".pdf"))
  png_path <- file.path(out_dir, paste0(filename, ".png"))
  tryCatch(
    ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf, bg = "white"),
    error = function(e) ggsave(pdf_path, plot, width = width, height = height, device = pdf, bg = "white")
  )
  ggsave(png_path, plot, width = width, height = height, dpi = 300, bg = "white")
  cat("[SAVED]", filename, "(", width, "x", height, "in)\n")
}

save_heatmap <- function(ht, filename, width, height) {
  pdf_path <- file.path(out_dir, paste0(filename, ".pdf"))
  png_path <- file.path(out_dir, paste0(filename, ".png"))
  pdf_device(pdf_path, width = width, height = height)
  draw(ht)
  dev.off()
  png(png_path, width = width, height = height, units = "in", res = 300, bg = "white")
  draw(ht)
  dev.off()
  cat("[SAVED]", filename, "(", width, "x", height, "in)\n")
}

save_base <- function(plot_fn, filename, width, height) {
  pdf_path <- file.path(out_dir, paste0(filename, ".pdf"))
  png_path <- file.path(out_dir, paste0(filename, ".png"))
  pdf_device(pdf_path, width = width, height = height)
  plot_fn()
  dev.off()
  png(png_path, width = width, height = height, units = "in", res = 300, bg = "white")
  plot_fn()
  dev.off()
  cat("[SAVED]", filename, "(", width, "x", height, "in)\n")
}

cor_label <- function(x, y, method = "spearman") {
  ct <- cor.test(x, y, method = method)
  sprintf("Spearman r = %.2f\np = %.2e", ct$estimate, ct$p.value)
}

###############################################################################
## DATA LOADING
###############################################################################
log_info("Loading data files...")

## --- LFC matrix (FA genes x timepoints) ---
lfc_matrix <- read.csv(lfc_matrix_path, stringsAsFactors = FALSE, row.names = 1)
tp_cols <- intersect(c("X3h", "X12h", "X24h", "X3D", "X7D"), colnames(lfc_matrix))
if (length(tp_cols) == 0) stop("No timepoint columns found in LFC matrix")
lfc_matrix <- lfc_matrix[, tp_cols, drop = FALSE]
lfc_matrix <- lfc_matrix[apply(is.finite(as.matrix(lfc_matrix)), 1, all), , drop = FALSE]
colnames(lfc_matrix) <- recode(colnames(lfc_matrix),
                               "X3h" = "3h", "X12h" = "12h", "X24h" = "1DPI",
                               "X3D" = "3DPI", "X7D" = "7DPI")
log_info("LFC matrix: ", nrow(lfc_matrix), " genes x ", ncol(lfc_matrix), " timepoints")

## --- Spatial object + scores ---
spatial_obj <- readRDS(spatial_rds_path)
spatial_scores <- read.csv(spatial_scores_path, stringsAsFactors = FALSE)
rownames(spatial_scores) <- spatial_scores$spot_id

## --- Single-cell object ---
sc_obj <- readRDS(sc_rds_path)

## --- Other tables ---
neuron_fp_raw <- read.csv(neuron_fp_path, stringsAsFactors = FALSE)
augur_raw <- read.csv(augur_path, stringsAsFactors = FALSE)
pseudotime_raw <- read.csv(pseudotime_path, stringsAsFactors = FALSE)

## --- Metabolomics ---
metab_long_raw <- read.csv(metab_long_path, stringsAsFactors = FALSE)
metab_meta_raw <- read.csv(metab_meta_path, stringsAsFactors = FALSE)

## --- GSEA / cross-omics ---
gsea_all <- read.csv(gsea_path, stringsAsFactors = FALSE)
shared_pw <- read.csv(shared_pw_path, stringsAsFactors = FALSE)
axis_match <- read.csv(axis_match_path, stringsAsFactors = FALSE)
axis_table <- read.csv(axis_table_path, stringsAsFactors = FALSE)

###############################################################################
## SPATIAL PLOT DATA
###############################################################################
spatial_cells <- rownames(spatial_obj@meta.data)
common_spots <- intersect(spatial_cells, spatial_scores$spot_id)
if (length(common_spots) == 0) stop("No common spots between spatial Seurat and region scores")
log_info("Spatial common spots: ", length(common_spots), " / ", length(spatial_cells))

spatial_meta <- spatial_obj@meta.data[common_spots, , drop = FALSE]
spatial_meta$spot_id <- common_spots
spatial_meta$region <- spatial_scores[common_spots, "region"]
spatial_meta$Ferroptosis <- spatial_scores[common_spots, "Ferroptosis"]
spatial_meta$Ferroaging <- spatial_scores[common_spots, "Ferroaging"]
neuron_cols <- intersect(c("prop_NeuronsGABA", "prop_NeuronsGLUT"), colnames(spatial_meta))
if (length(neuron_cols) != 2)
  stop("Neuron proportion columns (prop_NeuronsGABA/prop_NeuronsGLUT) not found in spatial meta.data")
spatial_meta$neuron_prop <- spatial_meta[[neuron_cols[1]]] + spatial_meta[[neuron_cols[2]]]

spatial_coords <- so_spatial_coords_all(spatial_obj)
spatial_coords <- spatial_coords[spatial_coords$spot_id %in% common_spots, , drop = FALSE]
spatial_plot_df <- merge(spatial_coords,
                         spatial_meta[, c("spot_id", "region", "Ferroptosis", "Ferroaging", "neuron_prop")],
                         by = "spot_id", all.x = TRUE)
spatial_plot_df <- spatial_plot_df[!is.na(spatial_plot_df$Ferroaging), , drop = FALSE]
spatial_plot_df$sample <- factor(spatial_plot_df$sample)
log_info("Spatial plot df: ", nrow(spatial_plot_df), " spots across ",
         nlevels(spatial_plot_df$sample), " samples")

###############################################################################
## SINGLE-CELL UMAP + SCORES
###############################################################################
sc_umap <- so_embedding(sc_obj, "umap")
sc_meta_full <- sc_obj@meta.data
sc_common <- intersect(rownames(sc_meta_full), rownames(sc_umap))
sc_umap <- sc_umap[sc_common, , drop = FALSE]
sc_meta_full <- sc_meta_full[sc_common, , drop = FALSE]

required_sc_cols <- c("Celltypes", "Condition", "Ferroptosis_UCell", "Senescence_UCell",
                      "Ferroaging_UCell", "Ferrosenescence_UCell")
missing_sc <- setdiff(required_sc_cols, colnames(sc_meta_full))
if (length(missing_sc) > 0) stop("Missing SC meta columns: ", paste(missing_sc, collapse = ", "))

## Sat1 expression
sat1_vec <- so_fetch_gene(sc_obj, "Sat1", layer = "data")
names(sat1_vec) <- rownames(sc_obj@meta.data)

sc_meta <- data.frame(
  UMAP_1            = sc_umap[, 1],
  UMAP_2            = sc_umap[, 2],
  Celltypes         = sc_meta_full$Celltypes,
  Condition         = sc_meta_full$Condition,
  Ferroptosis       = sc_meta_full$Ferroptosis_UCell,
  Senescence        = sc_meta_full$Senescence_UCell,
  Ferroaging        = sc_meta_full$Ferroaging_UCell,
  Ferrosenescence   = sc_meta_full$Ferrosenescence_UCell,
  Sat1              = as.numeric(sat1_vec[sc_common]),
  stringsAsFactors  = FALSE,
  row.names         = sc_common
)
sc_meta <- sc_meta[complete.cases(sc_meta[, c("UMAP_1", "UMAP_2", "Ferroaging", "Sat1")]), , drop = FALSE]
sc_meta$Celltypes <- factor(sc_meta$Celltypes)
sc_meta$Condition <- factor(sc_meta$Condition, levels = c("Ctrl", "1DPI", "3DPI", "7DPI"))
log_info("SC plot df: ", nrow(sc_meta), " cells; ", nlevels(sc_meta$Celltypes), " cell types")

umap_df <- sc_meta[, c("UMAP_1", "UMAP_2", "Celltypes", "Condition", "Ferroaging")]
colnames(umap_df)[5] <- "Ferroaging_UCell"

###############################################################################
## METABOLOMICS DIFFERENTIAL (Young = 3wk vs Old = 59wk)
###############################################################################
metab_meta <- metab_meta_raw %>%
  filter(!is.na(Age), Age %in% c("3 weeks", "59 weeks")) %>%
  select(sample_id, Age) %>% distinct() %>%
  mutate(age_group = ifelse(Age == "3 weeks", "Young", "Old"))

metab_long <- metab_long_raw %>%
  inner_join(metab_meta %>% select(sample_id, age_group), by = "sample_id") %>%
  filter(abundance > 0)

metab_res <- metab_long %>%
  group_by(metabolite) %>%
  filter(n_distinct(age_group) == 2, n() >= 6) %>%
  summarise(
    mean_young = mean(abundance[age_group == "Young"], na.rm = TRUE),
    mean_old   = mean(abundance[age_group == "Old"], na.rm = TRUE),
    log2FC     = log2(mean_old / mean_young),
    sem_log2FC = sqrt((sd(abundance[age_group == "Old"], na.rm = TRUE) /
                       (mean(abundance[age_group == "Old"], na.rm = TRUE) * log(2) *
                        sqrt(sum(age_group == "Old"))))^2 +
                      (sd(abundance[age_group == "Young"], na.rm = TRUE) /
                       (mean(abundance[age_group == "Young"], na.rm = TRUE) * log(2) *
                        sqrt(sum(age_group == "Young"))))^2),
    pvalue     = t.test(abundance ~ age_group)$p.value,
    .groups    = "drop"
  ) %>%
  mutate(padj = p.adjust(pvalue, method = "BH"),
         sig  = case_when(padj < 0.001 ~ "***",
                          padj < 0.01  ~ "**",
                          padj < 0.05  ~ "*",
                          TRUE ~ ""))
metab_res <- metab_res[is.finite(metab_res$log2FC), , drop = FALSE]
log_info("Metabolomics differential: ", nrow(metab_res), " metabolites tested")

###############################################################################
## GSEA DATA
###############################################################################
gsea_all$timepoint <- recode(gsea_all$comparison,
                             "24h" = "1DPI", "3D" = "3DPI", "7D" = "7DPI",
                             .default = gsea_all$comparison)
gsea_all$timepoint <- factor(gsea_all$timepoint,
                             levels = c("3h", "12h", "1DPI", "3DPI", "7DPI"))
ferroaging_pathways <- c("Senescence", "Ferroaging", "Ferrosenescence", "Ferroptosis")
gsea_core <- gsea_all %>%
  filter(Description %in% ferroaging_pathways) %>%
  mutate(Description = factor(Description, levels = ferroaging_pathways),
         sig_label = case_when(p.adjust < 0.001 ~ "***",
                               p.adjust < 0.01  ~ "**",
                               p.adjust < 0.05  ~ "*",
                               p.adjust < 0.1   ~ ".",
                               TRUE ~ ""))
gsea_core$timepoint <- droplevels(gsea_core$timepoint)
if (nrow(gsea_core) == 0) stop("No GSEA rows match ferroaging pathways")
log_info("GSEA core: ", nrow(gsea_core), " rows; timepoints: ",
         paste(levels(gsea_core$timepoint), collapse = ","))

###############################################################################
## PSEUDOTIME (filter Inf)
###############################################################################
n_inf <- sum(is.infinite(pseudotime_raw$pseudotime))
if (n_inf > 0)
  log_warn("Filtering ", n_inf, " Inf pseudotime rows out of ", nrow(pseudotime_raw),
           " (un-rooted cells with no trajectory assignment)")
pseudotime <- pseudotime_raw[is.finite(pseudotime_raw$pseudotime), , drop = FALSE]
if (nrow(pseudotime) == 0) stop("No finite pseudotime values remain after filtering Inf")
pseudotime$Condition <- factor(pseudotime$Condition, levels = c("Ctrl", "1DPI", "3DPI", "7DPI"))
log_info("Pseudotime: ", nrow(pseudotime), " finite cells (dropped ", n_inf, " Inf)")

###############################################################################
## FIGURE 1A — Spatial Ferroaging score
###############################################################################
log_info("Building Fig1A (spatial ferroaging)...")
tissue_df <- spatial_plot_df[spatial_plot_df$tissue == 1, , drop = FALSE]
nontissue_df <- spatial_plot_df[spatial_plot_df$tissue == 0, , drop = FALSE]

f1a <- ggplot() +
  geom_point(data = nontissue_df, aes(x = imagecol, y = -imagerow),
             color = "grey95", size = 1.5, shape = 16) +
  geom_point(data = tissue_df, aes(x = imagecol, y = -imagerow, color = Ferroaging),
             size = 2.5, alpha = 0.9, shape = 16) +
  scale_color_viridis_c(option = "magma", name = "Ferroaging\nscore") +
  facet_wrap(~ sample, nrow = 1) +
  coord_equal() +
  labs(title = "Spatial distribution of ferroaging score") +
  theme_luxe() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        axis.title = element_blank(), axis.line = element_blank(),
        strip.text = element_text(size = 10, face = "bold"),
        panel.background = element_rect(fill = "white", color = "grey90"),
        panel.grid = element_blank())
save_single(f1a, "Fig1A_spatial_ferroaging", width = 5, height = 4)

###############################################################################
## FIGURE 1B — Single-cell UMAP Ferroaging
###############################################################################
log_info("Building Fig1B (UMAP ferroaging)...")
f1b <- ggplot(umap_df, aes(UMAP_1, UMAP_2, color = Ferroaging_UCell)) +
  geom_point(size = 1.5, alpha = 0.85, shape = 16) +
  scale_color_viridis_c(option = "magma", name = "Ferroaging\nscore") +
  labs(title = "Single-cell ferroaging score (UCell)",
       x = "UMAP 1", y = "UMAP 2") +
  theme_luxe() +
  theme(axis.text = element_blank(), axis.ticks = element_blank())
save_single(f1b, "Fig1B_umap_ferroaging", width = 5, height = 4.5)

###############################################################################
## FIGURE 1C — Top12 metabolite log2FC waterfall
###############################################################################
log_info("Building Fig1C (metabolite waterfall)...")
metab_top12 <- metab_res %>%
  arrange(padj, desc(abs(log2FC))) %>%
  head(12) %>%
  arrange(log2FC) %>%
  mutate(metabolite = factor(metabolite, levels = metabolite),
         direction = ifelse(log2FC > 0, "Up", "Down"))

f1c <- ggplot(metab_top12, aes(x = metabolite, y = log2FC, fill = direction)) +
  geom_col(width = 0.7, color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = log2FC - sem_log2FC, ymax = log2FC + sem_log2FC),
                width = 0.2, linewidth = 0.4) +
  geom_text(aes(y = log2FC + ifelse(log2FC > 0, sem_log2FC + 0.05, -(sem_log2FC + 0.05)),
                label = sig),
            vjust = 0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Down" = "#2166AC", "Up" = "#B2182B"), guide = "none") +
  coord_flip() +
  labs(title = "Top 12 differentially abundant metabolites",
       x = "", y = expression("log"[2]*" FC (59w / 3w)")) +
  theme_luxe()
save_single(f1c, "Fig1C_metab_waterfall", width = 5, height = 4)

###############################################################################
## FIGURE 1D — GSEA NES heatmap (FA pathways x timepoints)
###############################################################################
log_info("Building Fig1D (GSEA heatmap)...")
nes_range <- max(abs(gsea_core$NES), na.rm = TRUE)
nes_lim <- ceiling(nes_range)
f1d <- ggplot(gsea_core, aes(x = timepoint, y = Description, fill = NES)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = sprintf("%.2f%s", NES, sig_label)), size = 4, fontface = "bold") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-nes_lim, nes_lim),
                       name = "NES", oob = scales::squish) +
  labs(title = "Ferroaging pathway GSEA (NES)",
       x = "Timepoint (vs Ctrl)", y = "") +
  theme_luxe() +
  theme(axis.text.x = element_text(angle = 0), panel.grid = element_blank())
save_single(f1d, "Fig1D_gsea_heatmap", width = 5, height = 3)

###############################################################################
## FIGURE 2A — FA gene log2FC ridge density
###############################################################################
log_info("Building Fig2A (ridge density)...")
tp_levels <- c("3h", "12h", "1DPI", "3DPI", "7DPI")
tp_present <- intersect(tp_levels, colnames(lfc_matrix))
lfc_long <- lfc_matrix[, tp_present, drop = FALSE] %>%
  rownames_to_column("gene") %>%
  pivot_longer(cols = all_of(tp_present), names_to = "tp", values_to = "log2FC") %>%
  mutate(tp = factor(tp, levels = tp_present))

f2a <- ggplot(lfc_long, aes(x = log2FC, y = tp, fill = after_stat(x))) +
  ggridges::geom_density_ridges_gradient(scale = 2.5, rel_min_height = 0.01,
                                         alpha = 0.85, bandwidth = 0.005) +
  scale_fill_gradientn(colors = pal_div_luxe, name = "log2FC", guide = "none") +
  ggridges::theme_ridges(font_family = font_family) +
  theme_luxe() +
  labs(title = paste0("FA-", nrow(lfc_matrix), " gene log2FC distribution across timepoints"),
       x = expression("log"[2]*" Fold Change"), y = "") +
  theme(axis.text.y = element_text(size = rel(1.1)))
save_single(f2a, "Fig2A_ridge_density", width = 6, height = 3.5)

###############################################################################
## FIGURE 2B — FA gene log2FC heatmap (ComplexHeatmap, kmeans=3)
###############################################################################
log_info("Building Fig2B (FA heatmap)...")
mat <- as.matrix(lfc_matrix[, tp_present, drop = FALSE])
set.seed(42)
km_clusters <- kmeans(mat, centers = 3)$cluster
max_abs <- max(abs(mat), na.rm = TRUE)
col_fun <- circlize::colorRamp2(c(-max_abs, 0, max_abs), c("#2166AC", "white", "#B2182B"))

tp_pal <- pal_cond_luxe[tp_present]
tp_pal <- tp_pal[!is.na(tp_pal)]
if (length(tp_pal) < length(tp_present))
  tp_pal <- setNames(pal_luxe[seq_along(tp_present)], tp_present)

col_anno <- HeatmapAnnotation(
  Timepoint = factor(colnames(mat), levels = tp_present),
  col = list(Timepoint = tp_pal),
  show_annotation_name = FALSE,
  annotation_name_gp = gpar(fontsize = 9, fontfamily = font_family),
  simple_anno_size = unit(0.4, "cm"))
row_anno <- rowAnnotation(
  Cluster = factor(km_clusters),
  col = list(Cluster = setNames(pal_luxe[1:3], sort(unique(km_clusters)))),
  show_annotation_name = FALSE,
  simple_anno_size = unit(0.4, "cm"))

ht2 <- Heatmap(mat, name = "log2FC", col = col_fun,
  row_split = factor(km_clusters, levels = sort(unique(km_clusters))),
  row_title = "Cluster %s", row_title_gp = gpar(fontsize = 10, fontface = "bold", fontfamily = font_family),
  cluster_columns = FALSE, cluster_row_slices = TRUE,
  show_row_names = FALSE, show_column_names = TRUE,
  column_names_gp = gpar(fontsize = 10, fontfamily = font_family),
  top_annotation = col_anno, left_annotation = row_anno,
  column_title = "Timepoints",
  column_title_gp = gpar(fontsize = 11, fontface = "bold", fontfamily = font_family),
  border = TRUE, use_raster = TRUE, raster_quality = 4,
  heatmap_legend_param = list(title_gp = gpar(fontsize = 10, fontface = "bold", fontfamily = font_family),
                              labels_gp = gpar(fontsize = 9, fontfamily = font_family)))
save_heatmap(ht2, "Fig2B_heatmap", width = 5, height = 6)

###############################################################################
## FIGURE 3A — Spatial Ferroptosis score
###############################################################################
log_info("Building Fig3A (spatial ferroptosis)...")
f3a <- ggplot() +
  geom_point(data = nontissue_df, aes(x = imagecol, y = -imagerow),
             color = "grey95", size = 1.5, shape = 16) +
  geom_point(data = tissue_df, aes(x = imagecol, y = -imagerow, color = Ferroptosis),
             size = 2.5, alpha = 0.9, shape = 16) +
  scale_color_viridis_c(option = "viridis", name = "Ferroptosis\nscore") +
  facet_wrap(~ sample, nrow = 1) +
  coord_equal() +
  labs(title = "Spatial ferroptosis score") +
  theme_luxe() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        axis.title = element_blank(), axis.line = element_blank(),
        strip.text = element_text(size = 10, face = "bold"),
        panel.background = element_rect(fill = "white", color = "grey90"),
        panel.grid = element_blank())
save_single(f3a, "Fig3A_spatial_ferroptosis", width = 5, height = 4)

###############################################################################
## FIGURE 3B — Spatial region annotation
###############################################################################
log_info("Building Fig3B (spatial region)...")
f3b <- ggplot() +
  geom_point(data = nontissue_df, aes(x = imagecol, y = -imagerow),
             color = "grey95", size = 1.5, shape = 16) +
  geom_point(data = tissue_df, aes(x = imagecol, y = -imagerow, color = region),
             size = 2.5, alpha = 0.9, shape = 16) +
  scale_color_manual(values = pal_region_luxe, name = "Region") +
  facet_wrap(~ sample, nrow = 1) +
  coord_equal() +
  labs(title = "Tissue region annotation") +
  theme_luxe() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        axis.title = element_blank(), axis.line = element_blank(),
        strip.text = element_text(size = 10, face = "bold"),
        panel.background = element_rect(fill = "white", color = "grey90"),
        panel.grid = element_blank())
save_single(f3b, "Fig3B_spatial_region", width = 5, height = 4)

###############################################################################
## FIGURE 3C — Region vs Ferroaging violin + box + jitter (manual sig brackets)
###############################################################################
log_info("Building Fig3C (violin region)...")
violin_data <- tissue_df[is.finite(tissue_df$Ferroaging), , drop = FALSE]
violin_data$region <- factor(violin_data$region, levels = c("Healthy", "Penumbra", "Other"))

sig_pairs <- list(c("Healthy", "Penumbra"), c("Healthy", "Other"), c("Penumbra", "Other"))
bracket_y <- max(violin_data$Ferroaging, na.rm = TRUE)
step <- diff(range(violin_data$Ferroaging, na.rm = TRUE)) * 0.08
brackets <- do.call(rbind, lapply(seq_along(sig_pairs), function(i) {
  pr <- sig_pairs[[i]]
  d1 <- violin_data$Ferroaging[violin_data$region == pr[1]]
  d2 <- violin_data$Ferroaging[violin_data$region == pr[2]]
  pval <- wilcox.test(d1, d2)$p.value
  lab <- if (pval < 0.001) "***" else if (pval < 0.01) "**" else if (pval < 0.05) "*" else "ns"
  data.frame(x1 = pr[1], x2 = pr[2], y = bracket_y + i * step, label = lab,
             stringsAsFactors = FALSE)
}))

f3c <- ggplot(violin_data, aes(region, Ferroaging, fill = region)) +
  geom_violin(alpha = 0.5, trim = TRUE, color = "black", linewidth = 0.4) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.9, color = "black", linewidth = 0.4) +
  geom_jitter(width = 0.1, size = 0.6, alpha = 0.4, color = "grey40") +
  geom_segment(data = brackets, aes(x = x1, xend = x2, y = y, yend = y),
               inherit.aes = FALSE, color = "black", linewidth = 0.4) +
  geom_segment(data = brackets, aes(x = x1, xend = x1, y = y, yend = y - step * 0.3),
               inherit.aes = FALSE, color = "black", linewidth = 0.4) +
  geom_segment(data = brackets, aes(x = x2, xend = x2, y = y, yend = y - step * 0.3),
               inherit.aes = FALSE, color = "black", linewidth = 0.4) +
  geom_text(data = brackets, aes(x = (as.numeric(factor(x1, levels = levels(violin_data$region))) +
                                       as.numeric(factor(x2, levels = levels(violin_data$region)))) / 2,
                                 y = y + step * 0.15, label = label),
            inherit.aes = FALSE, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = pal_region_luxe, guide = "none") +
  labs(title = "Ferroaging score by tissue region",
       x = "Region", y = "Ferroaging score") +
  theme_luxe() +
  theme(legend.position = "none")
save_single(f3c, "Fig3C_violin_region", width = 4, height = 4)

###############################################################################
## FIGURE 3D — Neuron proportion vs Ferroptosis correlation (marginal density)
###############################################################################
log_info("Building Fig3D (neuron correlation)...")
cor_data <- neuron_fp_raw[is.finite(neuron_fp_raw$neuron_prop) & is.finite(neuron_fp_raw$fp_score), , drop = FALSE]
if (nrow(cor_data) == 0) stop("No finite rows in neuron_fp data")
cor_lab_3d <- cor_label(cor_data$neuron_prop, cor_data$fp_score)

f3d_base <- ggplot(cor_data, aes(neuron_prop, fp_score)) +
  geom_point(alpha = 0.4, size = 1.0, color = "#6A9FB5") +
  geom_smooth(method = "lm", color = "#D4675E", se = TRUE, fill = "grey85", linewidth = 0.8) +
  annotate("text", x = Inf, y = Inf, label = cor_lab_3d, hjust = 1.1, vjust = 1.5, size = 4) +
  labs(title = "Neuron proportion vs ferroptosis score",
       x = "Neuron proportion", y = "Ferroptosis score") +
  theme_luxe()
f3d <- ggMarginal(f3d_base, type = "density", fill = "#6A9FB5", color = "black")
save_single(f3d, "Fig3D_neuron_correlation", width = 4.5, height = 4)

###############################################################################
## FIGURE 4A — UMAP cell types
###############################################################################
log_info("Building Fig4A (UMAP cell types)...")
ct_levels <- names(sort(table(sc_meta$Celltypes), decreasing = TRUE))
sc_meta$Celltypes <- factor(sc_meta$Celltypes, levels = ct_levels)
umap_df$Celltypes <- sc_meta$Celltypes[match(rownames(umap_df), rownames(sc_meta))]

ct_pal <- setNames(rep_len(pal_luxe, length(ct_levels)), ct_levels)
f4a <- ggplot(umap_df, aes(UMAP_1, UMAP_2, color = Celltypes)) +
  geom_point(size = 1.5, alpha = 0.85, shape = 16) +
  scale_color_manual(values = ct_pal, name = "Cell type") +
  labs(title = "Single-cell UMAP: cell types", x = "UMAP 1", y = "UMAP 2") +
  theme_luxe() +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        legend.position = "right") +
  guides(color = guide_legend(override.aes = list(size = 2.5, alpha = 1),
                              ncol = 1, keywidth = 0.5, keyheight = 0.5))
save_single(f4a, "Fig4A_umap_celltypes", width = 5, height = 4.5)

###############################################################################
## FIGURE 4B — Ferroaging ridge by cell type
###############################################################################
log_info("Building Fig4B (ridge celltype ferroaging)...")
ct_median <- sc_meta %>%
  group_by(Celltypes) %>%
  summarise(med = median(Ferroaging, na.rm = TRUE), .groups = "drop") %>%
  arrange(med)
sc_meta$Celltypes_ridge <- factor(sc_meta$Celltypes, levels = ct_median$Celltypes)

f4b <- ggplot(sc_meta, aes(Ferroaging, Celltypes_ridge, fill = after_stat(x))) +
  ggridges::geom_density_ridges_gradient(scale = 1.8, rel_min_height = 0.01,
                                         alpha = 0.85) +
  scale_fill_gradientn(colors = pal_div_luxe, name = "Ferroaging\nscore", guide = "none") +
  ggridges::theme_ridges(font_family = font_family) +
  theme_luxe() +
  labs(title = "Ferroaging score distribution by cell type",
       x = "Ferroaging score (UCell)", y = "") +
  theme(axis.text.y = element_text(size = rel(1.0)))
save_single(f4b, "Fig4B_ridge_celltype", width = 5, height = 4)

###############################################################################
## FIGURE 4C — Ferroptosis vs Senescence correlation
###############################################################################
log_info("Building Fig4C (corr fp vs sen)...")
cor_lab_4c <- cor_label(sc_meta$Ferroptosis, sc_meta$Senescence)
f4c <- ggplot(sc_meta, aes(Ferroptosis, Senescence)) +
  geom_point(size = 0.8, alpha = 0.5, color = "#6A9FB5") +
  geom_smooth(method = "lm", color = "#D4675E", se = FALSE, linewidth = 0.8) +
  annotate("text", x = Inf, y = Inf, label = cor_lab_4c, hjust = 1.1, vjust = 1.5, size = 4) +
  labs(x = "Ferroptosis score", y = "Senescence score") +
  theme_luxe(base_size = 11)
save_single(f4c, "Fig4C_corr_fp_sen", width = 3.5, height = 3.5)

###############################################################################
## FIGURE 4D — Ferroptosis vs Sat1 correlation
###############################################################################
log_info("Building Fig4D (corr fp vs Sat1)...")
cor_lab_4d <- cor_label(sc_meta$Ferroptosis, sc_meta$Sat1)
f4d <- ggplot(sc_meta, aes(Ferroptosis, Sat1)) +
  geom_point(size = 0.8, alpha = 0.5, color = "#6A9FB5") +
  geom_smooth(method = "lm", color = "#D4675E", se = FALSE, linewidth = 0.8) +
  annotate("text", x = Inf, y = Inf, label = cor_lab_4d, hjust = 1.1, vjust = 1.5, size = 4) +
  labs(x = "Ferroptosis score", y = "Sat1 expression") +
  theme_luxe(base_size = 11)
save_single(f4d, "Fig4D_corr_fp_sat1", width = 3.5, height = 3.5)

###############################################################################
## FIGURE 4E — Senescence vs Sat1 correlation
###############################################################################
log_info("Building Fig4E (corr sen vs Sat1)...")
cor_lab_4e <- cor_label(sc_meta$Senescence, sc_meta$Sat1)
f4e <- ggplot(sc_meta, aes(Senescence, Sat1)) +
  geom_point(size = 0.8, alpha = 0.5, color = "#6A9FB5") +
  geom_smooth(method = "lm", color = "#D4675E", se = FALSE, linewidth = 0.8) +
  annotate("text", x = Inf, y = Inf, label = cor_lab_4e, hjust = 1.1, vjust = 1.5, size = 4) +
  labs(x = "Senescence score", y = "Sat1 expression") +
  theme_luxe(base_size = 11)
save_single(f4e, "Fig4E_corr_sen_sat1", width = 3.5, height = 3.5)

###############################################################################
## FIGURE 4F — Augur AUC lollipop
###############################################################################
log_info("Building Fig4F (Augur lollipop)...")
augur_df <- augur_raw %>%
  mutate(comparison = factor(comparison, levels = c("1DPI", "3DPI", "7DPI")))
f4f <- ggplot(augur_df, aes(reorder(cell_type, AUC), AUC, color = comparison)) +
  geom_segment(aes(xend = cell_type, yend = 0.5), color = "grey70", linewidth = 0.5) +
  geom_point(size = 3.5) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", linewidth = 0.4) +
  facet_wrap(~ comparison, ncol = 1, scales = "free_y") +
  scale_color_manual(values = pal_cond_luxe, guide = "none") +
  coord_flip() +
  labs(title = "Augur cell-type perturbation priority",
       x = "", y = "AUC (baseline = 0.5)") +
  theme_luxe() +
  theme(strip.text = element_text(size = 10, face = "bold"),
        axis.text.y = element_text(size = 9))
save_single(f4f, "Fig4F_augur_lollipop", width = 5, height = 4)

###############################################################################
## FIGURE 4G — Pseudotime density by Condition
###############################################################################
log_info("Building Fig4G (pseudotime density)...")
f4g <- ggplot(pseudotime, aes(pseudotime, fill = Condition, color = Condition)) +
  geom_density(alpha = 0.4, linewidth = 0.6, adjust = 1.2) +
  scale_fill_manual(values = pal_cond_luxe) +
  scale_color_manual(values = pal_cond_luxe) +
  labs(title = "Pseudotime distribution by condition",
       x = "Pseudotime", y = "Density") +
  theme_luxe()
save_single(f4g, "Fig4G_pseudotime_density", width = 5, height = 3.5)

###############################################################################
## FIGURE 4H — UCell along pseudotime LOESS
###############################################################################
log_info("Building Fig4H (pseudotime LOESS)...")
pseudotime_long <- pseudotime %>%
  select(pseudotime, Condition,
         Ferroptosis_UCell, Senescence_UCell,
         Ferrosenescence_UCell, Ferroaging_UCell) %>%
  pivot_longer(cols = ends_with("_UCell"), names_to = "Signature", values_to = "Score") %>%
  mutate(Signature = gsub("_UCell$", "", Signature),
         Signature = factor(Signature, levels = names(pal_sig_luxe)))

f4h <- ggplot(pseudotime_long, aes(pseudotime, Score, color = Signature)) +
  geom_point(size = 0.5, alpha = 0.2) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2, linewidth = 1.0, span = 0.4) +
  scale_color_manual(values = pal_sig_luxe) +
  labs(title = "Iron-aging signatures along pseudotime",
       x = "Pseudotime", y = "UCell score") +
  theme_luxe()
save_single(f4h, "Fig4H_pseudotime_loess", width = 6, height = 4)

###############################################################################
## FIGURE 5A — SAT1-polyamine axis metabolite waterfall
###############################################################################
log_info("Building Fig5A (SAT1 waterfall)...")
sat1_axis <- axis_table %>%
  filter(axis_name == "SAT1-polyamine") %>%
  mutate(direction = recode(direction_aging, "UP" = "Up", "DOWN" = "Down"),
         sig_label = sig_stars(p_adj_aging)) %>%
  arrange(log2FC_aging) %>%
  mutate(display_name = factor(display_name, levels = display_name))
if (nrow(sat1_axis) == 0) stop("No SAT1-polyamine axis rows found in axis_table")

f5a <- ggplot(sat1_axis, aes(display_name, log2FC_aging, fill = direction)) +
  geom_col(width = 0.7, color = "black", linewidth = 0.3) +
  geom_text(aes(y = log2FC_aging + ifelse(log2FC_aging > 0, 0.05, -0.05),
                label = sig_label),
            vjust = 0.5, size = 4, fontface = "bold") +
  coord_flip() +
  scale_fill_manual(values = c("Down" = "#2166AC", "Up" = "#B2182B"), guide = "none") +
  labs(title = "SAT1-polyamine axis metabolites",
       x = "", y = expression("log"[2]*" FC (59w / 3w)")) +
  theme_luxe() +
  theme(axis.text.y = element_text(size = 10))
save_single(f5a, "Fig5A_sat1_waterfall", width = 5, height = 4)

###############################################################################
## FIGURE 5B — Driver gene — KEGG pathway chord diagram
###############################################################################
log_info("Building Fig5B (chord diagram)...")
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
if (is.null(edges_df) || nrow(edges_df) == 0)
  stop("No driver-gene / KEGG pathway edges found")
edges_df <- edges_df[order(-edges_df$score), , drop = FALSE]
if (nrow(edges_df) > 30) edges_df <- head(edges_df, 30)
log_info("Chord edges: ", nrow(edges_df), " (top 30 by cross_omics_score)")

all_sectors <- unique(c(as.character(edges_df$from), as.character(edges_df$to)))
grid_colors <- character(length(all_sectors))
names(grid_colors) <- all_sectors
for (s in all_sectors) {
  if (s %in% drivers) {
    ev <- drv_evidence[s]
    grid_colors[s] <- pal_evidence_luxe[ifelse(is.na(ev) || !(ev %in% names(pal_evidence_luxe)), "Weak", ev)]
  } else {
    grid_colors[s] <- "#BDBDBD"
  }
}

chord_plot_fn <- function() {
  circos.clear()
  circos.par(gap.after = 2, start.degree = 90, clock.wise = FALSE)
  chordDiagram(edges_df[, c("from", "to", "score")],
               grid.col = grid_colors, transparency = 0.30,
               annotationTrack = "grid", preAllocateTracks = 1)
  circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    xlim <- get.cell.meta.data("xlim"); ylim <- get.cell.meta.data("ylim")
    sector.name <- get.cell.meta.data("sector.index")
    circos.text(mean(xlim), ylim[1], sector.name, facing = "clockwise",
                niceFacing = TRUE, adj = c(0, 0.5), cex = 0.55, col = "black",
                fontfamily = font_family)
  }, bg.border = NA)
  title(main = "Driver gene - KEGG pathway network", cex.main = 1.2, font.main = 2)
  circos.clear()
}
save_base(chord_plot_fn, "Fig5B_chord_diagram", width = 6, height = 6)

###############################################################################
## DONE
###############################################################################
log_info("All 20 independent figures generated successfully in: ", out_dir)
log_info("Output files: Fig1A-Fig1D, Fig2A-Fig2B, Fig3A-Fig3D, Fig4A-Fig4H, Fig5A-Fig5B (PDF + PNG each)")
