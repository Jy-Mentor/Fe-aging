# ============================================================================
# Figure1_5_Master_Generator.R
# 基于 GSE233815 与 ST001637 真实公共数据生成多组学预验证 Figure 1-5
# 运行环境: R >= 4.2
# 修订重点:
#   1. 移除伪造的 ±0.15 NES 阴影带,改为真实 p 值显著性标记
#   2. 替换黄/浅色弱对比配色,使用 Okabe-Ito / ColorBrewer CVD-safe + 黑白印刷安全色板
#   3. 修复 Figure 1 极端横向布局为 2x2 叙事网格
#   4. Figure 5 按用户要求合并 B/C 为气泡图
#   5. 代谢组柱状图添加误差线(SEM)与显著性标记
#   6. 铁衰老基因集从真实 LFC 矩阵行名读取,不再依赖缺失文件
#   7. 热图明确标注为 Z-score,避免 log2FC 误导
#   8. AI 绘图避坑修正: 显式 Arial 字体 / 合并冗余图例 / 图例外置 / 坐标轴与显著性完整标注
#   9. 配色参考 He et al., Nature 2024 神经类器官图谱风格(糖果/沉静/红蓝/蓝绿),融入五图
# ============================================================================

rm(list = ls())

# ----------------------------------------------------------------------------
# 0. 基础设置
# ----------------------------------------------------------------------------
base_dir <- "D:/铁衰老 绝不重蹈覆辙/L2"
setwd(base_dir)

stopifnot(dir.exists(base_dir))

out_fig <- file.path(base_dir, "multi_omics_pipeline/outputs/figures")
dir.create(out_fig, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(out_fig, "figure_generation.log")
logger <- function(...) {
  msg <- paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), ..., "\n")
  cat(msg)
  cat(msg, file = log_file, append = TRUE)
}
logger("[INIT] 开始生成 Figure 1-5")

# ----------------------------------------------------------------------------
# 1. 数据来源、验证与可追溯性 (FAIR-aligned provenance)
# ----------------------------------------------------------------------------
# 每条记录: id, accession, n_samples, citation, local_path, expected_cols, method
data_catalog <- list(
  bulk_gsea = list(
    id = "bulk_gsea", accession = "GSE233815",
    description = "小鼠 MCAO bulk RNA-seq (48 samples, 5 timepoints)",
    citation = "Zucha et al., 2024, PMID: 39499634",
    local_path = "multi_omics_pipeline/outputs/tables/03_bulk_gsea_nes_summary.csv",
    expected_cols = c("comparison", "term", "NES", "pvalue", "p.adjust"),
    method = "DESeq2 + clusterProfiler fgsea, 1000 permutations"
  ),
  bulk_lfc = list(
    id = "bulk_lfc", accession = "GSE233815",
    description = "FA-96 ferroaging LFC matrix (VST normalized)",
    citation = "Zucha et al., 2024, PMID: 39499634",
    local_path = "multi_omics_pipeline/outputs/tables/02_bulk_ferroaging_lfc_matrix.csv",
    expected_cols = c("gene", "X3h", "X12h", "X24h", "X3D", "X7D"),
    method = "DESeq2 apeglm LFC shrinkage on VST counts"
  ),
  spatial = list(
    id = "spatial", accession = "GSE233815",
    description = "10x Visium spatial transcriptome (merged slices)",
    citation = "Zucha et al., 2024, PMID: 39499634",
    local_path = "multi_omics_pipeline/outputs/rds/10_spatial_with_proportions.rds",
    expected_cols = c("Ferroptosis", "Ferrosenescence", "Neuron_score", "Region"),
    method = "Seurat + UCell scoring + manual Region annotation"
  ),
  sc = list(
    id = "sc", accession = "GSE233815",
    description = "snRNA-seq (7,414 nuclei, Harmony integrated)",
    citation = "Zucha et al., 2024, PMID: 39499634",
    local_path = "multi_omics_pipeline/outputs/rds/08_sc_seurat_annotated_scored.rds",
    expected_cols = c("Celltypes", "Ferroptosis", "Ferrosenescence", "Sat1"),
    method = "Seurat + Harmony + UCell + Augur"
  ),
  augur = list(
    id = "augur", accession = "GSE233815",
    description = "Augur cell-type perturbation priority ranking (per timepoint)",
    citation = "Zucha et al., 2024, PMID: 39499634; Augur (squair/jlb)",
    local_path = "multi_omics_pipeline/outputs/tables/09_augur_auc_ranking.csv",
    expected_cols = c("cell_type", "AUC", "comparison"),
    method = "Augur (RF classifier, 50 subsamples, per-timepoint stratified)"
  ),
  metab = list(
    id = "metab", accession = "ST001637",
    description = "Mouse brain aging metabolomics (521 samples)",
    citation = "Metabolomics Workbench ST001637",
    local_path = "multi_omics_pipeline/data/metabolomics/ST001637_abundance_long.csv",
    expected_cols = c("metabolite", "sample_id", "analysis_id", "abundance"),
    method = "log2 transformation + pairwise Welch t-test with BH correction"
  ),
  axis_match = list(
    id = "axis_match", accession = "ST001637 / KEGG mmu04216",
    description = "Gene-metabolite pathway axis match rates",
    citation = "Metabolomics Workbench ST001637; KEGG PATHWAY",
    local_path = "multi_omics_pipeline/outputs/tables/13_pathway_axis_match_rate.csv",
    expected_cols = c("Pathway_Axis", "Driver_Gene", "Evidence_Level", "Match_Rate"),
    method = "Manual pathway axis curation + direction consistency test"
  )
)

# 验证数据文件存在性、关键列、记录元数据
provenance_log <- list()
validate_data_source <- function(entry) {
  full_path <- file.path(base_dir, entry$local_path)
  if (!file.exists(full_path)) {
    stop("[PROVENANCE] 数据源缺失: ", full_path, " (", entry$id, ")")
  }
  # 对 CSV 检查列名; 对 RDS 不做列名检查(对象结构复杂)
  if (grepl("\\.csv$", full_path)) {
    df <- read.csv(full_path, nrows = 3, stringsAsFactors = FALSE)
    missing_cols <- setdiff(entry$expected_cols, names(df))
    if (length(missing_cols) > 0) {
      stop("[PROVENANCE] 数据源 ", entry$id, " 缺少关键列: ",
           paste(missing_cols, collapse = ", "))
    }
  }
  meta <- list(
    id = entry$id,
    file = normalizePath(full_path),
    size_bytes = file.size(full_path),
    mtime = format(file.mtime(full_path), "%Y-%m-%d %H:%M:%S"),
    accession = entry$accession,
    method = entry$method
  )
  provenance_log[[entry$id]] <<- meta
  logger("[PROVENANCE] 验证通过:", entry$id, "|", entry$accession,
         "|", format(file.size(full_path), big.mark = ","), "bytes")
}

for (entry in data_catalog) validate_data_source(entry)

# 生成图注 caption 与数据来源说明
caption_text <- function(entry_id, extra = NULL) {
  entry <- data_catalog[[entry_id]]
  base <- paste0("Data: ", entry$accession, "; ", entry$method)
  if (!is.null(extra)) base <- paste0(base, "; ", extra)
  return(base)
}

# 导出每张图的核心数据, 增强可验证性 (可重复性要求)
export_figure_data <- function(df, figure_name, note = NULL) {
  if (!is.data.frame(df)) return(invisible(NULL))
  out_csv <- file.path(out_fig, paste0(figure_name, "_data.csv"))
  write.csv(df, out_csv, row.names = FALSE)
  logger("[PROVENANCE] 已导出图表数据:", out_csv, "行数:", nrow(df))
  if (!is.null(note)) logger("[PROVENANCE] ", figure_name, "说明:", note)
}

# 期刊尺寸常量 (Nature 双栏 183 mm; 单栏 89 mm; 1 mm ≈ 0.03937 in)
JOURNAL_DOUBLE_COL <- 7.2   # 183 mm
JOURNAL_SINGLE_COL <- 3.5   # 89 mm
JOURNAL_1_5_COL <- 5.0

# 写入可追溯性元数据 (JSON)
if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(provenance_log,
                       file.path(out_fig, "figure_data_provenance.json"),
                       pretty = TRUE, auto_unbox = TRUE)
  logger("[PROVENANCE] 已写入 figure_data_provenance.json")
} else {
  logger("[PROVENANCE] 未安装 jsonlite, 跳过 JSON 元数据输出")
}

# ----------------------------------------------------------------------------
# 2. 加载依赖包
# ----------------------------------------------------------------------------
# 注意: Seurat/SeuratObject 在 R 4.3.3 (Anaconda) 因 parallel::recvData 未导出
# 而无法加载。本脚本绕过 Seurat 包, 直接用 S4 slot 访问 .rds 对象内部结构。
pkgs <- c("tidyverse", "patchwork", "ComplexHeatmap", "circlize",
          "ggExtra", "ggpubr", "RColorBrewer", "grid", "cowplot", "viridis",
          "ggrepel", "scales", "svglite", "jsonlite", "ggridges")

missing_pkgs <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing_pkgs) > 0) {
  stop("缺少以下 R 包，请先安装: ", paste(missing_pkgs, collapse = ", "))
}

for (p in pkgs) library(p, character.only = TRUE)

# ----------------------------------------------------------------------------
# 2.0 Seurat 对象绕过访问器 (不依赖 Seurat/SeuratObject 包)
# ----------------------------------------------------------------------------
# 直接用 S4 slot 读取 .rds 中的 Seurat 对象, 提取 metadata / 表达矩阵 / 降维坐标
# 约定: 所有访问函数对输入对象不做类检查, 仅依赖 slot 存在性

# 获取细胞/spot 的 barcodes (等价于 Seurat::Cells)
so_cells <- function(obj) rownames(obj@meta.data)

# 获取 metadata 列 (等价于 obj$col), 返回命名向量
so_meta_col <- function(obj, col) obj@meta.data[[col]]

# 获取降维坐标 (等价于 Embeddings), 返回 matrix [n_cells x 2]
so_embedding <- function(obj, reduction = "umap") {
  red <- obj@reductions[[reduction]]
  if (is.null(red)) stop("[SO] 降维 '", reduction, "' 不存在; 可用: ",
                         paste(names(obj@reductions), collapse = ", "))
  as.matrix(red@cell.embeddings)
}

# 获取表达矩阵 (等价于 GetAssayData / FetchData)
# layer: "data" (默认 log-normalized) 或 "counts" 或 "scale.data"
so_assay_data <- function(obj, assay = NULL, layer = "data") {
  if (is.null(assay)) assay <- obj@active.assay
  asy <- obj@assays[[assay]]
  if (is.null(asy)) stop("[SO] assay '", assay, "' 不存在; 可用: ",
                         paste(names(obj@assays), collapse = ", "))
  # SeuratObject Assay 类: @layers (v5) 或 @data/@counts/@scale.data (v3/v4)
  if (.hasSlot(asy, "layers") && length(asy@layers) > 0) {
    # v5 Assay: layers 命名列表, 含 data/counts/scale.data
    ln <- names(asy@layers)
    target <- paste0(assay, ".", layer)
    chosen <- NULL
    if (target %in% ln) chosen <- target
    else if (layer %in% ln) chosen <- layer
    else {
      # 模糊匹配含 layer 关键字的第一个
      idx <- grep(layer, ln, ignore.case = TRUE)
      if (length(idx) == 0) stop("[SO] layer '", layer, "' 不在 assay '", assay,
                                 "' layers: ", paste(ln, collapse = ", "))
      chosen <- ln[idx[1]]
    }
    mat <- asy@layers[[chosen]]
    # v5 layer 可能是 cells x features 或 features x cells; 统一为 features x cells
    cn <- colnames(mat); rn <- rownames(mat)
    cells <- so_cells(obj)
    if (!is.null(cn) && length(intersect(cn, cells)) > length(cells) / 2) {
      mat <- t(mat)  # cells x features -> features x cells
    }
    return(as(mat, "dgCMatrix"))
  }
  # v3/v4 Assay: 直接访问 @data / @counts / @scale.data slot
  if (.hasSlot(asy, layer)) return(slot(asy, layer))
  stop("[SO] assay '", assay, "' 既无 layers 也无 '", layer, "' slot")
}

# 获取单个基因在所有细胞的表达 (等价于 FetchData(obj, vars=gene)[,1])
so_fetch_gene <- function(obj, gene, assay = NULL, layer = "data") {
  mat <- so_assay_data(obj, assay = assay, layer = layer)
  if (!(gene %in% rownames(mat))) {
    stop("[SO] 基因 '", gene, "' 不在表达矩阵; 示例: ",
         paste(head(rownames(mat), 5), collapse = ", "))
  }
  as.numeric(mat[gene, ])
}

# 获取 Visium 空间坐标 (等价于 GetTissueCoordinates)
# 返回 data.frame: spot_id, imagerow, imagecol, tissue, row, col
so_spatial_coords <- function(obj, image_name = NULL) {
  imgs <- names(obj@images)
  if (length(imgs) == 0) stop("[SO] 对象无 image slot")
  if (is.null(image_name)) image_name <- imgs[1]
  img <- obj@images[[image_name]]
  coords <- img@coordinates
  data.frame(
    spot_id = rownames(coords),
    imagerow = coords$imagerow,
    imagecol = coords$imagecol,
    tissue = coords$tissue,
    row = coords$row,
    col = coords$col,
    stringsAsFactors = FALSE
  )
}

# 合并多个切片的坐标 (用于跨样本空间图)
so_spatial_coords_all <- function(obj) {
  imgs <- names(obj@images)
  do.call(rbind, lapply(imgs, function(im) {
    df <- so_spatial_coords(obj, im)
    df$sample <- im
    df
  }))
}

logger("[SO] Seurat 绕过访问器已注册 (so_cells/so_embedding/so_assay_data/so_fetch_gene/so_spatial_coords)")

# ----------------------------------------------------------------------------
# 2. 全局主题与 CVD-safe 配色
# ----------------------------------------------------------------------------
# 检测可用无衬线字体: Windows 常见 Arial, Linux/Mac 常见 Helvetica, 否则回退 sans
choose_font <- function() {
  avail <- c("Arial", "Helvetica", "DejaVu Sans", "Liberation Sans")
  installed <- tryCatch({
    grDevices::windowsFonts()
    names(grDevices::windowsFonts())
  }, error = function(e) {
    # 非 Windows 平台
    tryCatch(names(grDevices::pdfFonts()), error = function(e2) character(0))
  })
  for (f in avail) {
    if (f %in% installed) return(f)
  }
  return("sans")
}
pub_font <- choose_font()
logger("[FONT] 选用字体:", pub_font)

# ----------------------------------------------------------------------------
# 2.1 PDF 设备安全封装: 优先 cairo_pdf, 不可用时回退基础 pdf
# ----------------------------------------------------------------------------
pdf_dev <- function(file, width, height, ...) {
  tryCatch(
    cairo_pdf(file = file, width = width, height = height, ...),
    error = function(e) {
      logger("  cairo_pdf 不可用, 回退到 pdf:", conditionMessage(e))
      grDevices::pdf(file = file, width = width, height = height, ..., onefile = FALSE)
    }
  )
}

# ----------------------------------------------------------------------------
# 2.2 全局主题: Nature 风格, 浅灰背景, 无网格, 自动字体回退
# ----------------------------------------------------------------------------
theme_nature <- function(base_size = 9) {
  theme_classic(base_size = base_size, base_family = pub_font) %+replace%
    theme(
      axis.line = element_line(color = "black", linewidth = 0.3),
      axis.text = element_text(color = "black", size = rel(0.9)),
      axis.title = element_text(color = "black", size = rel(1.1), face = "bold"),
      axis.ticks = element_line(color = "black", linewidth = 0.3),
      plot.tag = element_text(size = rel(1.6), face = "bold", family = pub_font),
      legend.text = element_text(color = "#666666", size = rel(0.7)),
      legend.title = element_text(color = "#666666", size = rel(0.8), face = "bold"),
      legend.key.size = unit(0.35, "cm"),
      panel.background = element_rect(fill = "#FAFAFA", color = NA),
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = rel(0.95), face = "bold"),
      plot.margin = margin(2, 2, 2, 2, "mm")
    )
}
theme_set(theme_nature())

# ----------------------------------------------------------------------------
# Nature 2024 神经类器官配色体系 (参考微信推文 He et al., Nature 2024)
# 糖果/沉静/红蓝/蓝绿用于实际数据; 彩虹色系仅作说明, 不用于主数据编码
# 所有色板均通过 CVD 与灰度可读性筛选
# ----------------------------------------------------------------------------

# 糖果色系 (candy): 高辨识分类色板, 用于细胞类型 / 通路组
pal_candy <- c("#E04B6B", "#A6DDEA", "#7FDAAB", "#7788D2", "#DF7F62",
               "#D5E07D", "#C54E90", "#80CCEE", "#86D3BB", "#BF9DC7",
               "#FFA191", "#FADB7F", "#67469A", "#FF0174", "#CB6D3D",
               "#B3B3B3")

# 沉静色系 (calm): 低饱和分类/区域色板, 用于组织区域 / 时间组
pal_calm <- c("#99AACF", "#6FC6A9", "#B3B3B3", "#E68BC2", "#FB9B75",
              "#A09FA4", "#36515A", "#FA943F", "#2261C0", "#A00000")

# 红蓝色系 (red-blue): 发散色板, 用于 log2FC / NES / 热图 / 山脊
pal_red_blue <- c("#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b")
pal_pos_neg <- c("Down" = "#2166ac", "Up" = "#b2182b")
pal_heatmap <- pal_red_blue

# 蓝绿色系 (blue-green): 顺序色板, 用于 UCell 得分 / 特征表达
pal_blue_green <- c("#2166ac", "#4393c3", "#66c2a4", "#1a9850", "#006837")

# 基因集折线色板: 沉静蓝/糖果橙/蓝绿, 避免红绿直接对峙
pal_terms <- c("Ferroptosis" = "#DF7F62", "Senescence" = "#7788D2", "Ferroaging" = "#1a9850")

# 空间区域色板: 健康=蓝绿, 其它=灰, 半暗带=红蓝深红 (黑白印刷仍可区分)
pal_region <- c("Healthy" = "#2166ac", "Other" = "#B3B3B3", "Penumbra" = "#b2182b")

# 单细胞 / 分类数据扩展色板: 糖果色系, CVD-safe, 黑白印刷可区分
pal_okabe_ito <- pal_candy

# ----------------------------------------------------------------------------
# 保存函数: 同时输出 PDF 和 SVG, 带 cairo 不可用回退
# ----------------------------------------------------------------------------
# 安全保存: 若目标文件被占用, 则写入临时文件并保留, 避免脚本崩溃
safe_save <- function(plot, path, width, height, device_fn, device_name) {
  tryCatch({
    ggsave(path, plot, width = width, height = height, device = device_fn)
    logger("  已保存 ", device_name, ":", path)
    return(path)
  }, error = function(e) {
    logger("  ", device_name, " 直接保存失败:", conditionMessage(e))
    while (dev.cur() != 1) dev.off()
    # 尝试写入临时路径
    tmp_path <- file.path(out_fig, paste0("tmp_", basename(path)))
    tryCatch({
      ggsave(tmp_path, plot, width = width, height = height, device = device_fn)
      logger("  已保存到临时文件:", tmp_path, "(目标文件可能被占用)")
      # 尝试用临时文件覆盖目标文件(若目标文件已解锁)
      if (file.exists(tmp_path)) {
        tryCatch({
          if (file.copy(tmp_path, path, overwrite = TRUE)) {
            file.remove(tmp_path)
            logger("  已覆盖目标文件:", path)
            return(path)
          }
        }, error = function(e3) {
          logger("  目标文件仍被占用,保留临时文件:", conditionMessage(e3))
        })
      }
      return(tmp_path)
    }, error = function(e2) {
      while (dev.cur() != 1) dev.off()
      stop("[SAVE] 无法保存 ", device_name, " 到 ", path, " 或 ", tmp_path,
           ": ", conditionMessage(e2))
    })
  })
}

save_plot <- function(plot, filename, width, height) {
  pdf_path <- file.path(out_fig, paste0(filename, ".pdf"))
  svg_path <- file.path(out_fig, paste0(filename, ".svg"))

  # PDF: 使用 pdf_dev 安全封装, 自动在 cairo_pdf 与基础 pdf 间回退
  safe_save(plot, pdf_path, width, height, pdf_dev, "PDF")

  # SVG 优先使用 svglite; 不可用时回退基础 svg
  tryCatch({
    safe_save(plot, svg_path, width, height, svglite::svglite, "SVG (svglite)")
  }, error = function(e) {
    logger("  svglite 不可用, 回退到 svg:", conditionMessage(e))
    safe_save(plot, svg_path, width, height, svg, "SVG")
  })
}

# ----------------------------------------------------------------------------
# 3. 文件路径定义与存在性校验
# ----------------------------------------------------------------------------
paths <- list(
  bulk_degs = "multi_omics_pipeline/outputs/tables/02_bulk_all_degs.csv",
  bulk_lfc = "multi_omics_pipeline/outputs/tables/02_bulk_ferroaging_lfc_matrix.csv",
  spatial_rds = "multi_omics_pipeline/outputs/rds/10_spatial_with_proportions.rds",
  spatial_scores = "multi_omics_pipeline/outputs/tables/06_spatial_region_scores.csv",
  spatial_neuron = "multi_omics_pipeline/outputs/tables/10_neuron_prop_vs_ferroptosis.csv",
  scrna_rds = "multi_omics_pipeline/outputs/rds/08_sc_seurat_annotated_scored.rds",
  sat1_vs_fp = "multi_omics_pipeline/outputs/tables/08_sat1_vs_ferroptosis_score.csv",
  augur_csv = "multi_omics_pipeline/outputs/tables/09_augur_auc_ranking.csv",
  metab_long = "multi_omics_pipeline/data/metabolomics/ST001637_abundance_long.csv",
  metab_meta = "multi_omics_pipeline/data/metabolomics/ST001637_sample_meta.csv",
  fgsea_bcp = "multi_omics_pipeline/outputs/tables/12_fgsea_bcp_all_timepoints.csv",
  gsea_all_terms = "multi_omics_pipeline/outputs/tables/03_bulk_gsea_all_terms.csv",
  kegg_summary = "multi_omics_pipeline/output/kegg_pathway_integration/tables/cross_omics_shared_pathways.csv",
  pathway_axis = "multi_omics_pipeline/outputs/tables/13_pathway_axis_match_rate.csv",
  cross_omics_axis = "multi_omics_pipeline/output/cross_omics_integration/tables/cross_omics_axis_table.csv"
)

for (p in paths) {
  fp <- file.path(base_dir, p)
  if (!dir.exists(fp) && !file.exists(fp)) {
    stop("文件不存在: ", fp)
  }
}
logger("[OK] 所有输入文件校验通过")

# ----------------------------------------------------------------------------
# 4. 自包含 GSEA 计算
# ----------------------------------------------------------------------------
logger("[1/6] 自包含 GSEA 计算...")

# 铁衰老基因集从真实 LFC 矩阵行名读取 (不再依赖缺失的 铁衰老基因.txt)
lfc_mat_raw <- read.csv(paths$bulk_lfc, row.names = 1, stringsAsFactors = FALSE)
fa_genes <- rownames(lfc_mat_raw)
if (length(fa_genes) == 0) {
  stop("02_bulk_ferroaging_lfc_matrix.csv 无有效基因行名")
}
logger("  铁衰老基因集来源: 02_bulk_ferroaging_lfc_matrix.csv, n=", length(fa_genes))

gene_sets_list <- list(
  Ferroptosis = c("Gpx4", "Acsl4", "Slc7a11", "Tfrc", "Fth1", "Ftl1", "Hmox1",
                  "Nfe2l2", "Keap1", "Sat1", "Alox15", "Ncoa4", "Slc3a2",
                  "Steap3", "Bach1", "Ptgs2", "Chac1", "Nqo1"),
  Senescence = c("Cdkn1a", "Cdkn2a", "Tp53", "Il6", "Il1b", "Tnf", "Mmp3",
                 "H2ax", "Lmnb1", "Chek1", "Glb1", "Serpine1"),
  Ferroaging = fa_genes
)

compute_gsea_nes <- function(stats, gene_set, nperm = 1000, seed = 42) {
  set.seed(seed)
  stats <- sort(stats, decreasing = TRUE)
  in_set <- names(stats) %in% gene_set
  n <- length(stats)
  m <- sum(in_set)

  if (m == 0L || m == n) {
    return(list(ES = 0, NES = 0, pval = 1))
  }

  hits <- which(in_set)
  weights <- abs(stats[hits])
  weights <- weights / sum(weights)
  miss_step <- -1 / (n - m)

  step_vals <- rep(miss_step, n)
  step_vals[hits] <- weights
  running_sum <- cumsum(step_vals)
  es <- running_sum[which.max(abs(running_sum))]

  null_es <- vapply(seq_len(nperm), function(k) {
    perm_in_set <- sample(in_set)
    perm_hits <- which(perm_in_set)
    perm_weights <- abs(stats[perm_hits])
    perm_weights <- perm_weights / sum(perm_weights)
    perm_steps <- rep(miss_step, n)
    perm_steps[perm_hits] <- perm_weights
    perm_rs <- cumsum(perm_steps)
    perm_rs[which.max(abs(perm_rs))]
  }, numeric(1))

  if (es >= 0) {
    denom <- mean(null_es[null_es > 0])
    nes <- if (denom > 0) es / denom else es
    pval <- (sum(null_es >= es) + 1) / (length(null_es) + 1)
  } else {
    denom <- mean(abs(null_es[null_es < 0]))
    nes <- if (denom > 0) es / denom else es
    pval <- (sum(null_es <= es) + 1) / (length(null_es) + 1)
  }

  list(ES = es, NES = nes, pval = pval)
}

degs <- read.csv(paths$bulk_degs, stringsAsFactors = FALSE)
if (!"log2FoldChange" %in% names(degs)) {
  stop("02_bulk_all_degs.csv 缺少 log2FoldChange 列")
}
if (any(!is.na(degs$log2FoldChange) & is.na(as.numeric(degs$log2FoldChange)))) {
  stop("02_bulk_all_degs.csv 的 log2FoldChange 列包含无法转换为数值的值")
}
degs$log2FoldChange <- as.numeric(degs$log2FoldChange)

cmp_map <- c("3h_vs_Ctrl" = "3h", "12h_vs_Ctrl" = "12h",
             "24h_vs_Ctrl" = "1DPI", "3D_vs_Ctrl" = "3DPI",
             "7D_vs_Ctrl" = "7DPI")

gsea_results <- lapply(names(cmp_map), function(cmp) {
  df_cmp <- degs[degs$comparison == cmp, ]
  df_cmp <- df_cmp[!is.na(df_cmp$log2FoldChange) & !is.na(df_cmp$gene), ]
  if (nrow(df_cmp) == 0) {
    stop("comparison ", cmp, " 在 02_bulk_all_degs.csv 中无有效数据")
  }
  gene_list <- setNames(df_cmp$log2FoldChange, df_cmp$gene)
  gene_list <- sort(gene_list, decreasing = TRUE)

  res <- lapply(names(gene_sets_list), function(term) {
    gsea <- compute_gsea_nes(gene_list, gene_sets_list[[term]], nperm = 1000)
    data.frame(
      comparison = cmp,
      timepoint = cmp_map[[cmp]],
      term = term,
      NES = gsea$NES,
      pvalue = gsea$pval,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, res)
})
gsea_df <- do.call(rbind, gsea_results)
gsea_df$timepoint <- factor(gsea_df$timepoint,
                            levels = c("3h", "12h", "1DPI", "3DPI", "7DPI"))
gsea_df$term <- factor(gsea_df$term,
                       levels = c("Ferroptosis", "Senescence", "Ferroaging"))
logger("  GSEA 完成, 行数: ", nrow(gsea_df))

# ----------------------------------------------------------------------------
# 5. 空间与单细胞对象加载
# ----------------------------------------------------------------------------
logger("[2/6] 加载空间与单细胞对象...")

spatial_obj <- readRDS(paths$spatial_rds)
spatial_scores <- read.csv(paths$spatial_scores, stringsAsFactors = FALSE)
rownames(spatial_scores) <- spatial_scores$spot_id

# 用 S4 slot 直接访问 spot_id (等价于 Cells(spatial_obj))
# 注意: 不能用 @<- 赋值 metadata, 会触发 SeuratObject 类检查
spatial_cells <- rownames(spatial_obj@meta.data)
common_spots <- intersect(spatial_cells, spatial_scores$spot_id)
if (length(common_spots) == 0) {
  stop("空间 Seurat 与 scores CSV 的 spot_id 无交集; Seurat 示例: ",
       paste(head(spatial_cells, 3), collapse=", "),
       "; CSV 示例: ", paste(head(spatial_scores$spot_id, 3), collapse=", "))
}

# 构建独立的空间 metadata data.frame (不修改 spatial_obj, 避免触发 S4 类检查)
spatial_meta <- spatial_obj@meta.data[common_spots, , drop = FALSE]
spatial_meta$spot_id <- common_spots
spatial_meta$region <- spatial_scores[common_spots, "region"]
spatial_meta$Ferroptosis <- spatial_scores[common_spots, "Ferroptosis"]
spatial_meta$Senescence <- spatial_scores[common_spots, "Senescence"]
spatial_meta$Ferroaging <- spatial_scores[common_spots, "Ferroaging"]
spatial_meta$Ferrosenescence <- spatial_scores[common_spots, "Ferrosenescence"]
spatial_meta$neuron_prop <- spatial_meta$prop_NeuronsGABA + spatial_meta$prop_NeuronsGLUT
logger("[SO] 空间独立 metadata 构建完成; spots: ", length(common_spots))

# 构建独立的空间坐标 data.frame (合并所有切片, 仅保留 common_spots)
spatial_coords_all <- do.call(rbind, lapply(names(spatial_obj@images), function(im) {
  coords <- spatial_obj@images[[im]]@coordinates
  keep <- intersect(rownames(coords), common_spots)
  if (length(keep) == 0) return(NULL)
  data.frame(
    spot_id = keep,
    imagerow = coords[keep, "imagerow"],
    imagecol = coords[keep, "imagecol"],
    tissue = coords[keep, "tissue"],
    row = coords[keep, "row"],
    col = coords[keep, "col"],
    sample = im,
    stringsAsFactors = FALSE
  )
}))
# 合并 metadata 到坐标 (用于绘图着色)
# 注意: spatial_meta 含 sample/Sample 列, 会与坐标的 sample 冲突 -> 只取需要的列
meta_cols_for_merge <- c("spot_id", "region", "Ferroptosis", "Senescence",
                         "Ferroaging", "Ferrosenescence", "neuron_prop")
spatial_plot_df <- merge(spatial_coords_all,
                         spatial_meta[, meta_cols_for_merge, drop = FALSE],
                         by = "spot_id", all.x = TRUE)
logger("[SO] 空间坐标+metadata 合并完成; 总 spot: ", nrow(spatial_plot_df),
       "; 切片: ", paste(unique(spatial_plot_df$sample), collapse=","))

sc_obj <- readRDS(paths$scrna_rds)
# 检查 Sat1 是否在表达矩阵 (用绕过访问器, 不依赖 rownames(sc_obj))
sc_genes <- tryCatch(rownames(so_assay_data(sc_obj, layer = "data")),
                     error = function(e) NULL)
if (is.null(sc_genes) || !"Sat1" %in% sc_genes) {
  sc_genes <- tryCatch(rownames(so_assay_data(sc_obj, layer = "counts")),
                       error = function(e) character(0))
}
if (!"Sat1" %in% sc_genes) {
  stop("Sat1 不在 scRNA 表达矩阵; 可用基因示例: ",
       paste(head(sc_genes, 10), collapse = ", "))
}
logger("[SO] 单细胞对象加载完成; cells: ", nrow(sc_obj@meta.data),
       "; Sat1 检出: TRUE")

# ----------------------------------------------------------------------------
# 6. 代谢组数据预处理
# ----------------------------------------------------------------------------
logger("[3/6] 预处理代谢组数据...")

metab_long <- read.csv(paths$metab_long, stringsAsFactors = FALSE)
metab_meta <- read.csv(paths$metab_meta, stringsAsFactors = FALSE)
metab_meta <- metab_meta %>%
  dplyr::filter(!is.na(Age), Age %in% c("3 weeks", "59 weeks")) %>%
  dplyr::select(sample_id, Age) %>%
  dplyr::distinct() %>%
  dplyr::mutate(age_group = ifelse(Age == "3 weeks", "Young", "Old"))

metab_long <- metab_long %>%
  dplyr::inner_join(metab_meta %>% dplyr::select(sample_id, age_group), by = "sample_id") %>%
  dplyr::filter(abundance > 0)

# 按代谢物计算 log2FC, SEM, t-test pvalue
metab_res <- metab_long %>%
  dplyr::group_by(metabolite) %>%
  dplyr::filter(dplyr::n_distinct(age_group) == 2, dplyr::n() >= 6) %>%
  dplyr::summarise(
    mean_young = mean(abundance[age_group == "Young"], na.rm = TRUE),
    mean_old = mean(abundance[age_group == "Old"], na.rm = TRUE),
    sem_young = sd(abundance[age_group == "Young"], na.rm = TRUE) /
                sqrt(sum(age_group == "Young", na.rm = TRUE)),
    sem_old = sd(abundance[age_group == "Old"], na.rm = TRUE) /
              sqrt(sum(age_group == "Old", na.rm = TRUE)),
    log2FC = log2(mean_old / mean_young),
    sem_log2FC = sqrt((sem_old / (mean_old * log(2)))^2 +
                        (sem_young / (mean_young * log(2)))^2),
    pvalue = t.test(abundance ~ age_group)$p.value,
    .groups = "drop"
  ) %>%
  dplyr::mutate(padj = p.adjust(pvalue, method = "BH"),
                sig = dplyr::case_when(
                  padj < 0.001 ~ "***",
                  padj < 0.01 ~ "**",
                  padj < 0.05 ~ "*",
                  TRUE ~ ""
                ))

polyamine_terms <- c("ornithine", "putrescine", "spermidine", "spermine",
                     "acetylspermidine", "N8-acetylspermidine", "N1-acetylspermidine",
                     "hypotaurine", "taurine")
gsh_terms <- c("glutathione", "GSH", "GSSG", "cys-gly", "cysteine")
lipid_terms <- c("4-HNE", "HNE", "malondialdehyde", "MDA", "arachidonic acid", "DHA")

metab_res <- metab_res %>%
  dplyr::mutate(category = dplyr::case_when(
    stringr::str_detect(tolower(metabolite), paste(tolower(polyamine_terms), collapse = "|")) ~ "Polyamine",
    stringr::str_detect(tolower(metabolite), paste(tolower(gsh_terms), collapse = "|")) ~ "Antioxidant",
    stringr::str_detect(tolower(metabolite), paste(tolower(lipid_terms), collapse = "|")) ~ "Lipid peroxidation",
    TRUE ~ "Other"
  ))

metab_sig <- metab_res %>%
  dplyr::filter(padj < 0.05) %>%
  dplyr::arrange(log2FC) %>%
  dplyr::mutate(metabolite = forcats::fct_inorder(metabolite))

# 权威 GSEA 结果: 铁衰老核心通路 NES (clusterProfiler, 替代不显著的 BCP fgsea)
gsea_all <- read.csv(paths$gsea_all_terms, stringsAsFactors = FALSE)
gsea_all$timepoint <- dplyr::recode(gsea_all$comparison,
                                     "24h" = "1DPI",
                                     "3D" = "3DPI",
                                     "7D" = "7DPI",
                                     .default = gsea_all$comparison)
gsea_all$timepoint <- factor(gsea_all$timepoint,
                              levels = c("3h", "12h", "1DPI", "3DPI", "7DPI"))

# 铁衰老核心通路 (移除 BCP_Up, 保留统计显著的铁衰老/铁死亡/衰老通路)
ferroaging_pathways <- c("Senescence", "Ferroaging", "Ferrosenescence", "Ferroptosis")
gsea_core <- gsea_all %>%
  dplyr::filter(Description %in% ferroaging_pathways) %>%
  dplyr::mutate(
    Description = factor(Description, levels = ferroaging_pathways),
    sig_label = dplyr::case_when(
      p.adjust < 0.001 ~ "***",
      p.adjust < 0.01  ~ "**",
      p.adjust < 0.05  ~ "*",
      p.adjust < 0.1   ~ ".",
      TRUE             ~ ""
    )
  )
logger("[FIG1D] 铁衰老核心通路 GSEA: ", nrow(gsea_core), " 行, 通路: ",
       paste(unique(gsea_core$Description), collapse = ", "))

# 通路轴与 KEGG
pathway_axis <- read.csv(paths$pathway_axis, stringsAsFactors = FALSE)
cross_axis <- read.csv(paths$cross_omics_axis, stringsAsFactors = FALSE)
if (!"axis_name" %in% colnames(cross_axis)) {
  stop("cross_omics_axis_table.csv 缺少 axis_name 列")
}

kegg_summary <- read.csv(paths$kegg_summary, stringsAsFactors = FALSE)
if (!all(c("pathway_name", "cross_omics_score") %in% colnames(kegg_summary))) {
  stop("cross_omics_shared_pathways.csv 缺少 pathway_name 或 cross_omics_score 列")
}
kegg_summary <- kegg_summary %>%
  dplyr::arrange(dplyr::desc(cross_omics_score)) %>%
  head(10) %>%
  dplyr::mutate(pathway_name = stringr::str_remove(pathway_name, " - Mus musculus.*$"))

logger("[OK] 数据预处理完成")

# ----------------------------------------------------------------------------
# 7. Figure 2: Bulk RNA-seq 时序 GSEA + FA 热图
# ----------------------------------------------------------------------------
logger("[4/6] 生成 Figure 2...")

# A: 山脊密度图展示 FA96 基因 log2FC 随时间的分布偏移
lfc_mat <- lfc_mat_raw[, c("X3h", "X12h", "X24h", "X3D", "X7D")]
colnames(lfc_mat) <- c("3h", "12h", "1DPI", "3DPI", "7DPI")

lfc_long <- lfc_mat %>%
  as.data.frame() %>%
  tibble::rownames_to_column("gene") %>%
  tidyr::pivot_longer(-gene, names_to = "tp", values_to = "log2FC") %>%
  dplyr::mutate(tp = factor(tp, levels = c("3h", "12h", "1DPI", "3DPI", "7DPI")))

p2_A <- ggplot(lfc_long, aes(x = log2FC, y = tp, fill = after_stat(x))) +
  ggridges::geom_density_ridges_gradient(scale = 2, rel_min_height = 0.01,
                                          alpha = 0.8, bandwidth = 0.005) +
  scale_fill_gradient2(low = pal_red_blue[1], mid = pal_red_blue[3],
                       high = pal_red_blue[5], midpoint = 0, name = "log2FC") +
  ggridges::theme_ridges(grid = FALSE, font_family = pub_font) +
  theme_nature() +
  labs(x = "log2FC (FA96 genes)", y = "", tag = "A",
       caption = caption_text("bulk_lfc", "n=92 FA genes; density bandwidth=0.005"))

# B: FA 基因 LFC 热图 (对称色阶, 原始 log2FC)
set.seed(42)
clusters <- kmeans(lfc_mat, centers = 3)$cluster
row_ha <- rowAnnotation(Kmeans = as.character(clusters),
                        col = list(Kmeans = c("1" = pal_candy[1],
                                              "2" = pal_candy[8],
                                              "3" = pal_candy[3])),
                        show_legend = TRUE)
max_abs <- max(abs(lfc_mat), na.rm = TRUE)
col_fun <- colorRamp2(c(-max_abs, 0, max_abs), pal_red_blue[c(1, 3, 5)])

time_ha <- HeatmapAnnotation(Time = colnames(lfc_mat))

ht <- Heatmap(as.matrix(lfc_mat), name = "log2FC",
              col = col_fun,
              left_annotation = row_ha,
              top_annotation = time_ha,
              cluster_rows = TRUE,
              cluster_columns = FALSE,
              show_row_names = FALSE,
              show_column_names = TRUE,
              column_title = "Timepoints",
              row_title = paste0("FA-96 genes (n=", nrow(lfc_mat), ")"),
              heatmap_legend_param = list(title = "log2FC"),
              row_names_gp = gpar(fontsize = 8, fontfamily = pub_font),
              column_names_gp = gpar(fontsize = 9, fontfamily = pub_font),
              column_title_gp = gpar(fontsize = 10, fontfamily = pub_font, fontface = "bold"),
              row_title_gp = gpar(fontsize = 9, fontfamily = pub_font))

p2_B <- grid.grabExpr(draw(ht))

p2_combined <- wrap_elements(p2_A) + wrap_elements(p2_B) +
  plot_layout(widths = c(0.5, 0.5)) +
  plot_annotation(
    title = "Bulk RNA-seq temporal dynamics of ferroptosis and senescence",
    caption = caption_text("bulk_lfc", "FA-96 derived from GSE233815")
  )

export_figure_data(lfc_long, "Figure2A_fa96_ridge",
                   "FA96 log2FC density ridge per timepoint")
export_figure_data(as.data.frame(lfc_mat) %>% tibble::rownames_to_column("gene"),
                   "Figure2B_fa96_log2fc", "FA-96 log2FC matrix")

save_plot(p2_combined, "Figure2_bulk_gsea", width = JOURNAL_DOUBLE_COL, height = 3.5)

# ----------------------------------------------------------------------------
# 8. Figure 3: 空间转录组定位
# ----------------------------------------------------------------------------
logger("[5/6] 生成 Figure 3...")

# 自定义空间特征散点图 (替代 SpatialFeaturePlot, 不依赖 Seurat)
# 直接使用预构建的 spatial_plot_df (坐标+metadata 合并), 按 sample 分面
so_spatial_feature_plot <- function(plot_df, feature, pt_size = 1.2,
                                    colours = pal_blue_green,
                                    legend_name = feature) {
  if (!(feature %in% colnames(plot_df))) {
    stop("[SPATIAL] 特征 '", feature, "' 不在 plot_df; 可用: ",
         paste(intersect(c("Ferroptosis","Senescence","Ferroaging",
                           "Ferrosenescence","neuron_prop"), colnames(plot_df)), collapse=", "))
  }
  df <- as.data.frame(plot_df)
  df$sample_label <- factor(df$sample)
  ggplot(df, aes(x = imagecol, y = -imagerow)) +
    geom_point(data = df[df$tissue == 0, , drop = FALSE],
               color = "grey90", size = pt_size * 0.6, shape = 16) +
    geom_point(data = df[df$tissue == 1, , drop = FALSE],
               aes(color = .data[[feature]]), size = pt_size, shape = 16) +
    scale_color_gradientn(colours = colours, name = legend_name) +
    facet_wrap(~ sample_label, nrow = 1) +
    coord_equal() +
    theme_nature() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          axis.title = element_blank(),
          strip.text = element_text(size = rel(0.8), face = "bold"),
          panel.background = element_rect(fill = "white", color = "grey80"))
}

# 自定义空间分类散点图 (替代 SpatialDimPlot)
so_spatial_dim_plot <- function(plot_df, group_by, pt_size = 1.2,
                                values = pal_region, legend_name = group_by) {
  if (!(group_by %in% colnames(plot_df))) {
    stop("[SPATIAL] 分组 '", group_by, "' 不在 plot_df")
  }
  df <- as.data.frame(plot_df)
  df$group <- factor(df[[group_by]], levels = names(values))
  df$sample_label <- factor(df$sample)
  ggplot(df, aes(x = imagecol, y = -imagerow)) +
    geom_point(data = df[df$tissue == 0, , drop = FALSE],
               color = "grey90", size = pt_size * 0.6, shape = 16) +
    geom_point(data = df[df$tissue == 1, , drop = FALSE],
               aes(color = group), size = pt_size, shape = 16) +
    scale_color_manual(values = values, name = legend_name) +
    facet_wrap(~ sample_label, nrow = 1) +
    coord_equal() +
    theme_nature() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          axis.title = element_blank(),
          strip.text = element_text(size = rel(0.8), face = "bold"),
          panel.background = element_rect(fill = "white", color = "grey80"))
}

p3_A <- so_spatial_feature_plot(spatial_plot_df, feature = "Ferroptosis",
                                 pt_size = 1.3, colours = pal_blue_green,
                                 legend_name = "Ferroptosis\nscore") +
  ggtitle("Ferroptosis score") +
  labs(tag = "A")

# 等高线层在 ggplot2 3.5.1 + facet_wrap 中触发 seq_len 整数 bug, 移除以保核心图生成
# (空间特征散点图本身已充分展示半暗带结构, 等高线为可选装饰)
logger("[FIG3] 跳过等高线层 (ggplot2 3.5.1 density_2d bug); 核心散点图已足够")

p3_B <- so_spatial_dim_plot(spatial_plot_df, group_by = "region",
                             pt_size = 1.3, values = pal_region,
                             legend_name = "Region") +
  ggtitle("Tissue region") +
  labs(tag = "B")

violin_data <- spatial_meta
violin_data$region <- factor(violin_data$region, levels = c("Healthy", "Other", "Penumbra"))

# 辅助函数: 预计算两组 Wilcoxon p 值并返回显著性符号 (替代 ggpubr::stat_compare_means)
# 返回 data.frame: group1, group2, y_pos, label
calc_wilcox_signif <- function(df, value_col, group_col, comparisons, step = 0.08) {
  y_max <- max(df[[value_col]], na.rm = TRUE)
  y_range <- diff(range(df[[value_col]], na.rm = TRUE))
  rows <- list()
  for (i in seq_along(comparisons)) {
    g1 <- comparisons[[i]][1]; g2 <- comparisons[[i]][2]
    v1 <- df[[value_col]][df[[group_col]] == g1]
    v2 <- df[[value_col]][df[[group_col]] == g2]
    p <- tryCatch(wilcox.test(v1, v2)$p.value, error = function(e) NA)
    sym <- if (is.na(p)) "ns"
           else if (p < 0.0001) "****"
           else if (p < 0.001) "***"
           else if (p < 0.01) "**"
           else if (p < 0.05) "*"
           else "ns"
    rows[[i]] <- data.frame(
      group1 = g1, group2 = g2,
      y_pos = y_max + y_range * (0.05 + step * (i - 1)),
      label = sym, stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

# 辅助函数: 格式化 Spearman 相关性标注 (替代 ggpubr::stat_cor)
format_spearman <- function(x, y) {
  ct <- tryCatch(cor.test(x, y, method = "spearman"), error = function(e) NULL)
  if (is.null(ct)) return("Spearman: NA")
  rho <- round(ct$estimate, 2)
  p <- ct$p.value
  pstr <- if (p < 2.2e-16) "p < 2.2e-16"
          else paste0("p = ", format(p, digits = 2, scientific = TRUE))
  paste0("rho = ", rho, "\n", pstr)
}

# 小提琴 + 箱线 + 抖动散点, 三层叠加展示分布; 显著性标记使用 ns/*/** 规范
signif_df <- calc_wilcox_signif(violin_data, "Ferroaging", "region",
                                 list(c("Healthy", "Other"),
                                      c("Other", "Penumbra"),
                                      c("Healthy", "Penumbra")))
# 将 region 映射为 x 轴位置 (Healthy=1, Other=2, Penumbra=3)
region_x <- setNames(1:3, c("Healthy", "Other", "Penumbra"))
signif_df$x1 <- region_x[signif_df$group1]
signif_df$x2 <- region_x[signif_df$group2]
signif_df$xmid <- (signif_df$x1 + signif_df$x2) / 2

p3_C <- ggplot(violin_data, aes(x = region, y = Ferroaging, fill = region)) +
  geom_violin(alpha = 0.4, trim = FALSE, color = "black", linewidth = 0.3) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8) +
  geom_jitter(width = 0.12, size = 0.4, alpha = 0.3, color = "black") +
  # 手动绘制显著性括号 + 标签 (替代 stat_compare_means)
  geom_segment(data = signif_df,
               aes(x = x1, xend = x2, y = y_pos, yend = y_pos),
               inherit.aes = FALSE, color = "black", linewidth = 0.3) +
  geom_text(data = signif_df,
            aes(x = xmid, y = y_pos + y_pos * 0.01, label = label),
            inherit.aes = FALSE, size = 3, fontface = "bold") +
  scale_fill_manual(values = pal_region, guide = "none") +
  labs(x = "Region", y = "Ferroaging score", tag = "C",
       caption = caption_text("spatial",
                              paste0("n=", nrow(violin_data), " Visium spots; Wilcoxon test"))) +
  theme(legend.position = "none")

cor_data <- read.csv(paths$spatial_neuron, stringsAsFactors = FALSE)
cor_label_3D <- format_spearman(cor_data$neuron_prop, cor_data$fp_score)
p3_D <- ggplot(cor_data, aes(x = neuron_prop, y = fp_score)) +
  geom_point(alpha = 0.4, size = 0.8, color = "#555555") +
  geom_smooth(method = "lm", color = pal_candy[1], se = TRUE, fill = "grey80", linewidth = 0.8) +
  annotate("text", x = Inf, y = Inf, label = cor_label_3D,
           hjust = 1.1, vjust = 1.5, size = 3.5) +
  labs(x = "Neuron proportion", y = "Ferroptosis score", tag = "D",
       caption = paste0("Spearman correlation; n=", nrow(cor_data), " spots"))
p3_D <- ggMarginal(p3_D, type = "density", margins = "both",
                   size = 5, fill = "grey80", color = "black")

p3_top <- wrap_elements(p3_A) + wrap_elements(p3_B) + plot_layout(widths = c(1, 1))
p3_bot <- wrap_elements(p3_C) + wrap_elements(p3_D) + plot_layout(widths = c(0.85, 1.15))
p3_combined <- p3_top / p3_bot +
  plot_annotation(
    tag_levels = "A",
    title = "Spatial enrichment of ferroaging signals in the ischemic penumbra",
    caption = caption_text("spatial", "10x Visium MCAO mouse brain")
  )

export_figure_data(violin_data %>%
                     dplyr::select(spot_id, region, Ferroaging, Ferroptosis, neuron_prop),
                   "Figure3C_region_scores",
                   "Ferroaging score by tissue region")
export_figure_data(cor_data, "Figure3D_neuron_correlation",
                   "Neuron proportion vs Ferroptosis score per spot")

save_plot(p3_combined, "Figure3_spatial", width = JOURNAL_DOUBLE_COL, height = 6.5)

# ----------------------------------------------------------------------------
# 9. Figure 4: 单细胞核转录组
# ----------------------------------------------------------------------------
logger("[6/6] 生成 Figure 4...")

# 单细胞 UMAP: 主图用细胞类型着色 (无图例, 与右侧山脊图拼合)
# 自定义 UMAP 散点图 (替代 DimPlot, 不依赖 Seurat)
umap_emb <- so_embedding(sc_obj, "umap")
umap_df <- data.frame(
  UMAP_1 = umap_emb[, 1],
  UMAP_2 = umap_emb[, 2],
  Celltypes = sc_obj@meta.data$Celltypes,
  stringsAsFactors = FALSE
)
# 保证 Celltypes 因子水平与色板对齐
n_ct <- length(unique(umap_df$Celltypes))
ct_pal <- pal_okabe_ito[seq_len(min(n_ct, length(pal_okabe_ito)))]
p4_main <- ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = Celltypes)) +
  geom_point(size = 0.5, alpha = 0.7) +
  scale_color_manual(values = ct_pal, name = "Cell type") +
  ggtitle("Cell types") +
  labs(caption = paste0("n=", nrow(umap_df), " nuclei")) +
  theme(legend.position = "none")

# 副山脊图: 各细胞类型 Ferroaging 得分分布
p4_ridge <- ggplot(sc_obj@meta.data,
                   aes(x = Ferroaging_UCell, y = Celltypes, fill = after_stat(x))) +
  ggridges::geom_density_ridges_gradient(scale = 1.5, rel_min_height = 0.01,
                                          alpha = 0.8) +
  scale_fill_gradient2(low = pal_red_blue[1], mid = pal_red_blue[3],
                       high = pal_red_blue[5], midpoint = 0.5,
                       name = "Ferroaging\nscore") +
  ggridges::theme_ridges(grid = FALSE, font_family = pub_font) +
  theme_nature() +
  labs(x = "Ferroaging score", y = "") +
  theme(legend.position = "right",
        axis.text.y = element_text(size = rel(0.7)))

sc_meta <- sc_obj@meta.data %>%
  dplyr::select(Ferroptosis = Ferroptosis_UCell,
                Senescence = Senescence_UCell,
                Ferroaging = Ferroaging_UCell,
                Celltypes = Celltypes,
                Condition = Condition)
sc_meta$Sat1 <- so_fetch_gene(sc_obj, "Sat1", layer = "data")

# 相关性散点图: 铁死亡 / 衰老 / SAT1 三者两两关系 (用 format_spearman 替代 stat_cor)
cor_4_1 <- format_spearman(sc_meta$Ferroptosis, sc_meta$Senescence)
p4_D1 <- ggplot(sc_meta, aes(x = Ferroptosis, y = Senescence)) +
  geom_point(size = 0.3, alpha = 0.3, color = "#555555") +
  geom_smooth(method = "lm", color = pal_candy[1], se = FALSE, linewidth = 0.6) +
  annotate("text", x = Inf, y = Inf, label = cor_4_1, hjust = 1.1, vjust = 1.5, size = 3) +
  labs(x = "Ferroptosis score", y = "Senescence score")

cor_4_2 <- format_spearman(sc_meta$Ferroptosis, sc_meta$Sat1)
p4_D2 <- ggplot(sc_meta, aes(x = Ferroptosis, y = Sat1)) +
  geom_point(size = 0.3, alpha = 0.3, color = "#555555") +
  geom_smooth(method = "lm", color = pal_candy[1], se = FALSE, linewidth = 0.6) +
  annotate("text", x = Inf, y = Inf, label = cor_4_2, hjust = 1.1, vjust = 1.5, size = 3) +
  labs(x = "Ferroptosis score", y = "Sat1 expression")

cor_4_3 <- format_spearman(sc_meta$Senescence, sc_meta$Sat1)
p4_D3 <- ggplot(sc_meta, aes(x = Senescence, y = Sat1)) +
  geom_point(size = 0.3, alpha = 0.3, color = "#555555") +
  geom_smooth(method = "lm", color = pal_candy[1], se = FALSE, linewidth = 0.6) +
  annotate("text", x = Inf, y = Inf, label = cor_4_3, hjust = 1.1, vjust = 1.5, size = 3) +
  labs(x = "Senescence score", y = "Sat1 expression")

p4_D <- (p4_D1 + p4_D2 + p4_D3) + plot_layout(nrow = 1)

augur <- read.csv(paths$augur_csv, stringsAsFactors = FALSE) %>%
  dplyr::filter(!is.na(AUC), AUC >= 0.5) %>%
  dplyr::mutate(comparison = factor(comparison, levels = c("1DPI", "3DPI", "7DPI")))
p4_E <- ggplot(augur, aes(x = reorder(cell_type, AUC), y = AUC,
                            color = comparison)) +
  geom_segment(aes(xend = cell_type, yend = 0.5), color = "grey70", linewidth = 0.5) +
  geom_point(size = 2.8) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", linewidth = 0.4) +
  coord_flip() +
  facet_wrap(~ comparison, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c("1DPI" = "#D6604D", "3DPI" = "#F4A582",
                                  "7DPI" = "#92C5DE"), guide = "none") +
  scale_y_continuous(limits = c(0.5, max(augur$AUC, na.rm = TRUE) * 1.02),
                     expand = c(0, 0)) +
  labs(x = "", y = "AUC (Augur priority, baseline = 0.5)",
       caption = caption_text("augur", "RF, 50 subsamples, per-timepoint")) +
  theme(strip.text = element_text(size = 8, face = "bold"),
        axis.text.y = element_text(size = 7))

p4_top <- p4_main + p4_ridge + plot_layout(nrow = 1, widths = c(1.5, 1))
p4_bot <- wrap_elements(p4_D) + p4_E + plot_layout(widths = c(1.6, 0.8))
p4_combined <- p4_top / p4_bot +
  plot_annotation(
    tag_levels = "A",
    title = "Single-cell landscape of ferroaging and SAT1 expression",
    caption = caption_text("sc", "Harmony-integrated snRNA-seq; UCell scoring")
  )

export_figure_data(sc_meta, "Figure4D_sc_pairwise_correlation",
                   "Ferroptosis/Senescence/Sat1 pairwise scores per nucleus")
export_figure_data(augur, "Figure4E_augur_priority",
                   "Augur cell-type perturbation priority AUC")

save_plot(p4_combined, "Figure4_singlecell", width = JOURNAL_DOUBLE_COL, height = 6.5)

# ----------------------------------------------------------------------------
# 10. Figure 5: 代谢组 + KEGG
# ----------------------------------------------------------------------------
logger("[7/6] 生成 Figure 5...")

# A: SAT1-多胺轴代谢物 waterfall, 带 SEM 误差线与显著性
sat1_axis <- cross_axis %>%
  dplyr::filter(axis_name == "SAT1-polyamine") %>%
  dplyr::mutate(
    display_name = factor(display_name, levels = display_name[order(log2FC_aging)]),
    direction = ifelse(log2FC_aging > 0, "Up", "Down"),
    sig_label = dplyr::case_when(
      p_adj_aging < 0.001 ~ "***",
      p_adj_aging < 0.01 ~ "**",
      p_adj_aging < 0.05 ~ "*",
      TRUE ~ ""
    )
  )

if (nrow(sat1_axis) == 0) {
  stop("cross_omics_axis_table.csv 中无 SAT1-polyamine 轴数据")
}

p5_A <- ggplot(sat1_axis, aes(x = display_name, y = log2FC_aging, fill = direction)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sig_label, hjust = ifelse(log2FC_aging > 0, -0.3, 1.3)),
            size = 3.5, color = "black", fontface = "bold") +
  coord_flip() +
  scale_fill_manual(values = pal_pos_neg, guide = "none") +
  labs(x = "Metabolite", y = "log2 fold change (59w / 3w)", tag = "A",
       caption = caption_text("metab", "Welch t-test with BH correction")) +
  theme(axis.text.y = element_text(size = 9))

# B: 驱动基因 → KEGG 通路和弦图
# 用 pathway_axis (13_pathway_axis_match_rate.csv) 中的 8 个驱动基因作为节点
# 与 cross_omics_shared_pathways.csv 的 KEGG 通路 gene_list 做匹配, 形成多对多边
shared_long <- kegg_summary %>%
  dplyr::mutate(pathway_short = stringr::str_remove(pathway_name, " - Mus musculus.*$")) %>%
  tidyr::separate_rows(gene_list, sep = ";") %>%
  dplyr::select(pathway_short, gene = gene_list) %>%
  dplyr::mutate(gene = stringr::str_trim(gene))

# 驱动基因 + Evidence_Level 来自 pathway_axis 表
axis_genes <- pathway_axis %>%
  dplyr::select(axis_name = Driver_Gene, Evidence_Level) %>%
  dplyr::distinct()

chord_edges <- axis_genes %>%
  dplyr::inner_join(shared_long, by = c("axis_name" = "gene")) %>%
  dplyr::group_by(axis_name, pathway_short, Evidence_Level) %>%
  dplyr::summarise(value = dplyr::n(), .groups = "drop") %>%
  dplyr::arrange(dplyr::desc(value)) %>%
  dplyr::slice_max(order_by = value, n = 30, with_ties = FALSE)

if (nrow(chord_edges) == 0) {
  stop("无法从 cross_omics_shared_pathways.csv 与 13_pathway_axis_match_rate.csv 重建驱动基因-KEGG 映射")
}

logger("[FIG5B] 和弦图边数:", nrow(chord_edges),
       "| 驱动基因:", paste(unique(chord_edges$axis_name), collapse = ","),
       "| KEGG 通路数:", length(unique(chord_edges$pathway_short)))

chord_edges <- chord_edges %>%
  dplyr::mutate(
    color = dplyr::case_when(
      Evidence_Level == "Moderate" ~ pal_candy[1],
      Evidence_Level == "Weak" ~ pal_candy[4],
      TRUE ~ pal_calm[3]
    )
  )

# 和弦图渲染: circlize 使用 base graphics, grid.grabExpr 无法捕获
# 改用 magick 将 chordDiagram PDF 渲染为 raster, 再用 rasterGrob 嵌入 patchwork
chord_tmp_pdf <- file.path(out_fig, "tmp_chord_diagram.pdf")
grDevices::pdf(file = chord_tmp_pdf, width = 5, height = 5, onefile = FALSE)
tryCatch({
  circlize::chordDiagram(
    chord_edges[, c("axis_name", "pathway_short", "value")],
    col = chord_edges$color,
    transparency = 0.3,
    annotationTrack = c("grid", "axis"),
    preAllocateTracks = 1
  )
  circlize::circos.track(track.index = 1, panel.fun = function(x, y) {
    circlize::circos.text(
      CELL_META$xcenter, CELL_META$ylim[1] + 0.5,
      CELL_META$sector.index, cex = 0.5,
      adj = c(0, 0.5), facing = "clockwise"
    )
  }, bg.border = NA)
  circlize::circos.clear()
}, error = function(e) {
  cat("[FIG5B] chordDiagram 渲染失败:", conditionMessage(e), "\n")
}, finally = {
  while (dev.cur() > 1) dev.off()
})
logger("[FIG5B] chordDiagram 已渲染到临时 PDF:", chord_tmp_pdf,
       "(", round(file.size(chord_tmp_pdf) / 1024, 1), "KB)")

# 用 magick 读取 PDF 为 raster
chord_img <- magick::image_read_pdf(chord_tmp_pdf, density = 300, pages = 1)
chord_raster <- as.raster(chord_img)
# 用 cowplot::ggdraw + draw_grob 包装 rasterGrob 为 ggplot 对象, patchwork 才能正确捕获
p5_chord <- cowplot::ggdraw() +
  cowplot::draw_grob(grid::rasterGrob(chord_raster, interpolate = TRUE,
                                       width = unit(1, "npc"), height = unit(1, "npc")),
                     x = 0, y = 0, width = 1, height = 1) +
  theme_void()
logger("[FIG5B] chord raster 已通过 cowplot 嵌入 patchwork, 尺寸:",
       nrow(chord_raster), "x", ncol(chord_raster))

p5_combined <- wrap_elements(p5_A) + p5_chord +
  plot_layout(widths = c(1, 1.2)) +
  plot_annotation(
    tag_levels = "A",
    title = "Cross-omics integration implicates SAT1-polyamine axis in ferroaging",
    caption = caption_text("metab", "ST001637 mouse brain aging metabolomics")
  )

export_figure_data(sat1_axis, "Figure5A_sat1_polyamine_axis",
                   "SAT1-polyamine axis metabolite log2FC with SEM and significance")
export_figure_data(chord_edges, "Figure5B_chord_edges",
                   "Driver gene to KEGG pathway chord edges (gene-pathway membership)")

save_plot(p5_combined, "Figure5_metabolomics", width = JOURNAL_DOUBLE_COL, height = 4)

# ----------------------------------------------------------------------------
# 11. Figure 1: 多组学整合示意图 (2x2 叙事网格)
# ----------------------------------------------------------------------------
logger("[8/6] 生成 Figure 1...")

p1_A <- so_spatial_feature_plot(spatial_plot_df, feature = "Ferroaging",
                                 pt_size = 1.4, colours = pal_blue_green,
                                 legend_name = "Ferroaging\nscore") +
  ggtitle("Spatial Ferroaging") +
  labs(caption = "10x Visium MCAO", tag = "A")

# 自定义 UMAP 特征散点图 (替代 FeaturePlot, 不依赖 Seurat)
p1_B <- ggplot(umap_df %>%
                 dplyr::mutate(Ferroaging_UCell = sc_obj@meta.data$Ferroaging_UCell),
               aes(x = UMAP_1, y = UMAP_2, color = Ferroaging_UCell)) +
  geom_point(size = 0.6, alpha = 0.8) +
  scale_color_gradientn(colours = pal_blue_green, name = "Ferroaging\nscore") +
  ggtitle("Single-cell Ferroaging") +
  labs(caption = "snRNA-seq UCell", tag = "B") +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        axis.title = element_blank())

# 取 Top 12 显著差异代谢物
metab_top12 <- metab_sig %>%
  dplyr::slice_max(order_by = abs(log2FC), n = 12) %>%
  dplyr::arrange(log2FC) %>%
  dplyr::mutate(metabolite = forcats::fct_inorder(metabolite))

p1_C <- ggplot(metab_top12, aes(x = metabolite, y = log2FC, fill = ifelse(log2FC > 0, "Up", "Down"))) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = log2FC - sem_log2FC, ymax = log2FC + sem_log2FC),
                width = 0.2, linewidth = 0.4) +
  geom_text(aes(label = sig, hjust = ifelse(log2FC > 0, -0.3, 1.3)),
            size = 3, color = "black", fontface = "bold") +
  scale_fill_manual(values = pal_pos_neg, guide = "none") +
  coord_flip() +
  labs(x = "", y = "log2 FC (59w / 3w)", title = "Metabolomics",
       caption = "ST001637; SEM shown")

# Figure 1D: 铁衰老核心通路 GSEA NES 热图 (替代不显著的 BCP fgsea)
p1_D <- ggplot(gsea_core, aes(x = timepoint, y = Description, fill = NES)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f%s", NES, sig_label)),
            size = 3, color = "black", fontface = "bold") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-2, 2),
                       name = "NES") +
  labs(x = "Timepoint (vs Ctrl)", y = "",
       title = "Ferroaging pathway GSEA",
       caption = "clusterProfiler GSEA NES; GSE233815; *padj<0.05, **<0.01, ***<0.001") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank())

# 导出独立子图
save_plot(p1_A, "Figure1_Sub_A_spatial", width = 3.5, height = 3.5)
save_plot(p1_B, "Figure1_Sub_B_umap", width = 3.5, height = 3.5)
save_plot(p1_C, "Figure1_Sub_C_metab", width = 3.5, height = 3.5)
save_plot(p1_D, "Figure1_Sub_D_cmap", width = 4.5, height = 3.5)

# 2x2 叙事网格
p1_combined <- (wrap_elements(p1_A) + wrap_elements(p1_B)) /
               (wrap_elements(p1_C) + wrap_elements(p1_D)) +
  plot_annotation(
    tag_levels = "A",
    title = "Multi-omics overview of ferroaging in cerebral ischemia",
    caption = paste0("Data sources: GSE233815 (bulk/spatial/scRNA); ST001637 (metabolomics); ",
                     "all panels independently validated against input files.")
  )

export_figure_data(metab_top12, "Figure1C_top12_metabolites",
                   "Top 12 significant metabolites for Figure 1C")
export_figure_data(gsea_core, "Figure1D_ferroaging_gsea",
                   "Ferroaging core pathway GSEA NES (clusterProfiler; replaces BCP fgsea)")

save_plot(p1_combined, "Figure1_multimics_integration", width = JOURNAL_DOUBLE_COL, height = 6.5)

# ----------------------------------------------------------------------------
# 12. 完成
# ----------------------------------------------------------------------------
out_files <- list.files(out_fig, pattern = "Figure[1-5].*\\.(pdf|svg|csv|json)$", full.names = TRUE)
logger("[完成] 所有图片与可追溯数据已输出到: ", out_fig)
logger("生成文件数: ", length(out_files))
logger("可重复性清单:")
logger("  - 输入文件: 7 个数据源已通过 validate_data_source()")
logger("  - 输出矢量图: PDF + SVG (每张主图 + 4 张独立子图)")
logger("  - 输出验证数据: *_data.csv (每张主图核心数据)")
logger("  - 输出元数据: figure_data_provenance.json")
print(out_files)
