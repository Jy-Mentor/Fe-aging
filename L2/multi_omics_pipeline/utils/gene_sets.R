# ============================================================================
# 基因集管理 (gene_sets.R)
# - 加载项目自有铁衰老基因集 (铁衰老基因.txt)
# - 人源基因 → 鼠源基因映射 (基于 biomaRt / 手工映射表)
# - 提供 BCP signature / ferroptosis / senescence 标准基因集
# ============================================================================

# ----------------------------------------------------------------------------
# 人源 → 鼠源基因名映射 (基于命名约定, 首字母大写)
# ----------------------------------------------------------------------------
map_human_to_mouse <- function(human_genes) {
  # 规则: 人类基因全大写 (GPX4), 小鼠基因首字母大写 (Gpx4)
  # 已是小鼠命名的基因保持不变
  mouse_genes <- vapply(human_genes, function(g) {
    if (nchar(g) == 0) return(g)
    # 全大写视为人类基因
    if (g == toupper(g)) {
      paste0(toupper(substr(g, 1, 1)), tolower(substr(g, 2, nchar(g))))
    } else {
      g
    }
  }, character(1))
  return(mouse_genes)
}

# ----------------------------------------------------------------------------
# 加载项目自有铁衰老基因集
# ----------------------------------------------------------------------------
load_ferroaging_genes <- function(cfg) {
  fa_file <- cfg$gene_sets$ferroaging_file
  if (is.null(fa_file) || !file.exists(fa_file)) {
    log_warn("Ferroaging gene file not found: ", fa_file,
             ". Using config gene_sets$ferroptosis instead.")
    return(cfg$gene_sets$ferroptosis)
  }
  genes <- readLines(fa_file, encoding = "UTF-8")
  genes <- trimws(genes)
  genes <- genes[nzchar(genes) & !startsWith(genes, "#")]
  log_info("Loaded ferroaging gene set: ", length(genes), " genes from ", fa_file)
  return(genes)
}

# ----------------------------------------------------------------------------
# 整合所有基因集 (统一接口)
# ----------------------------------------------------------------------------
build_gene_sets <- function(cfg, organism = "mouse") {
  ferroaging <- load_ferroaging_genes(cfg)
  if (organism == "mouse") {
    ferroaging <- map_human_to_mouse(ferroaging)
  }

  ferroptosis <- if (organism == "mouse") {
    cfg$gene_sets$ferroptosis
  } else {
    toupper(cfg$gene_sets$ferroptosis)
  }

  senescence <- if (organism == "mouse") {
    cfg$gene_sets$senescence
  } else {
    toupper(cfg$gene_sets$senescence)
  }

  bcp_up <- if (organism == "mouse") {
    cfg$gene_sets$bcp_up
  } else {
    toupper(cfg$gene_sets$bcp_up)
  }

  bcp_down <- if (organism == "mouse") {
    cfg$gene_sets$bcp_down
  } else {
    toupper(cfg$gene_sets$bcp_down)
  }

  # 细胞类型 marker
  celltype_markers <- list(
    Neuron         = if (organism == "mouse") cfg$gene_sets$neuron         else toupper(cfg$gene_sets$neuron),
    Astrocyte      = if (organism == "mouse") cfg$gene_sets$astrocyte      else toupper(cfg$gene_sets$astrocyte),
    Microglia      = if (organism == "microglia") cfg$gene_sets$microglia  else toupper(cfg$gene_sets$microglia),
    Oligodendrocyte= if (organism == "mouse") cfg$gene_sets$oligodendrocyte else toupper(cfg$gene_sets$oligodendrocyte),
    Endothelial    = if (organism == "mouse") cfg$gene_sets$endothelial    else toupper(cfg$gene_sets$endothelial),
    Pericyte       = if (organism == "mouse") cfg$gene_sets$pericyte       else toupper(cfg$gene_sets$pericyte)
  )

  stress <- if (organism == "mouse") cfg$gene_sets$stress else toupper(cfg$gene_sets$stress)
  infarct_core <- if (organism == "mouse") cfg$gene_sets$infarct_core else toupper(cfg$gene_sets$infarct_core)

  list(
    ferroaging      = ferroaging,
    ferroptosis     = ferroptosis,
    senescence      = senescence,
    bcp_up          = bcp_up,
    bcp_down        = bcp_down,
    ferrosenescence = unique(c(ferroptosis, senescence, ferroaging)),
    bcp_signature   = list(up = bcp_up, down = bcp_down),
    celltype_markers= celltype_markers,
    stress          = stress,
    infarct_core    = infarct_core
  )
}

# ----------------------------------------------------------------------------
# 验证基因集与表达矩阵的重叠
# ----------------------------------------------------------------------------
validate_gene_set_overlap <- function(gene_set, expr_genes, set_name = "gene_set") {
  avail <- intersect(gene_set, expr_genes)
  missing <- setdiff(gene_set, expr_genes)
  log_info(sprintf("[%s] Available: %d/%d (%.1f%%)",
                   set_name, length(avail), length(gene_set),
                   100 * length(avail) / max(length(gene_set), 1)))
  if (length(missing) > 0) {
    log_debug(sprintf("[%s] Missing genes: %s",
                      set_name, paste(head(missing, 10), collapse = ", ")))
  }
  invisible(list(available = avail, missing = missing))
}
