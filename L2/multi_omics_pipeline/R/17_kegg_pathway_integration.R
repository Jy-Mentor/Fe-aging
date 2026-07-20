# ===========================================================================
# 17_kegg_pathway_integration.R
# 铁衰老项目 - KEGG 通路水平跨组学整合
#
# 目标:
#   1. 通过 KEGG REST API 查询 FA-96 基因对应的 KEGG 通路
#   2. 通过 KEGG REST API 查询跨数据集一致代谢物 (10个) 对应的 KEGG 通路
#   3. 找出基因和代谢物的共享 KEGG 通路 (跨组学通路证据)
#   4. 可视化跨组学通路富集叠合
#
# 数据源:
#   - FA-96 基因集: L1/results/ferroaging_genes_96.csv (96 个人源基因符号)
#   - 跨数据集一致代谢物: output/metabolomics_cross_validation/tables/cross_dataset_consistent_metabolites.csv
#   - KEGG REST API: https://rest.kegg.jp
# ===========================================================================

suppressPackageStartupMessages({
  library(httr)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(RColorBrewer)
})

# ===========================================================================
# 0. 路径与输出配置
# ===========================================================================
BASE_DIR       <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
FA_GENE_CSV    <- "D:/铁衰老 绝不重蹈覆辙/L1/results/ferroaging_genes_96.csv"
CROSS_META_CSV <- file.path(BASE_DIR, "output",
                              "metabolomics_cross_validation", "tables",
                              "cross_dataset_consistent_metabolites.csv")

OUT_DIR <- file.path(BASE_DIR, "output", "kegg_pathway_integration")
FIG_DIR <- file.path(OUT_DIR, "figures")
TAB_DIR <- file.path(OUT_DIR, "tables")
CACHE_DIR <- file.path(OUT_DIR, "cache")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TAB_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(CACHE_DIR, showWarnings = FALSE, recursive = TRUE)

KEGG_BASE <- "https://rest.kegg.jp"

# ===========================================================================
# 1. KEGG REST API 辅助函数
# ===========================================================================
kegg_rest <- function(endpoint, cache_key = NULL, cache_dir = CACHE_DIR) {
  if (!is.null(cache_key)) {
    cache_file <- file.path(cache_dir, paste0(cache_key, ".txt"))
    if (file.exists(cache_file)) {
      return(readLines(cache_file, warn = FALSE))
    }
  }

  url <- paste0(KEGG_BASE, endpoint)
  message(sprintf("  [KEGG] GET %s", url))
  r <- tryCatch(
    httr::GET(url, httr::timeout(60)),
    error = function(e) {
      message(sprintf("  [KEGG] 错误: %s", conditionMessage(e)))
      return(NULL)
    }
  )

  if (is.null(r) || httr::status_code(r) != 200) {
    warning(sprintf("  [KEGG] %s 请求失败", endpoint))
    return(character(0))
  }

  content_text <- httr::content(r, "text", encoding = "UTF-8")
  lines <- strsplit(content_text, "\n")[[1]]

  if (!is.null(cache_key)) {
    cache_file <- file.path(cache_dir, paste0(cache_key, ".txt"))
    writeLines(lines, cache_file)
  }

  Sys.sleep(0.5)
  lines
}

# ===========================================================================
# 2. FA-96 基因 → KEGG 基因 ID 映射 (人源基因符号 → mmu 小鼠同源)
# ===========================================================================
map_genes_to_kegg <- function(gene_symbols) {
  message("\n[1] 映射 FA-96 基因到 KEGG 小鼠同源...")

  results <- data.frame(
    human_symbol = character(),
    kegg_gene_id = character(),
    kegg_description = character(),
    stringsAsFactors = FALSE
  )

  for (gene in gene_symbols) {
    cache_key <- paste0("gene_find_", gene)
    lines <- kegg_rest(paste0("/find/mmu/", gene), cache_key = cache_key)

    if (length(lines) == 0 || all(lines == "")) {
      message(sprintf("    %s: 未找到 KEGG 小鼠同源", gene))
      next
    }

    parsed_lines <- strsplit(lines, "\t")
    best_match <- NULL

    for (pl in parsed_lines) {
      if (length(pl) < 2) next
      kegg_id <- pl[1]
      description <- pl[2]

      is_exact <- grepl(paste0("^", gene, ","), description, ignore.case = TRUE) ||
                  grepl(paste0("^", gene, ";"), description, ignore.case = TRUE) ||
                  grepl(paste0("\\b", gene, "\\b"), description, ignore.case = TRUE)

      if (is.null(best_match) || is_exact) {
        best_match <- list(kegg_id = kegg_id, description = description, exact = is_exact)
        if (is_exact) break
      }
    }

    if (!is.null(best_match)) {
      results <- rbind(results, data.frame(
        human_symbol = gene,
        kegg_gene_id = best_match$kegg_id,
        kegg_description = best_match$description,
        stringsAsFactors = FALSE
      ))
      message(sprintf("    %s -> %s (%s)",
                      gene, best_match$kegg_id,
                      substr(best_match$description, 1, 50)))
    } else {
      message(sprintf("    %s: 无最佳匹配", gene))
    }
  }

  message(sprintf("\n  KEGG 基因映射: %d/%d (%.1f%%)",
                  nrow(results), length(gene_symbols),
                  100 * nrow(results) / length(gene_symbols)))
  results
}

# ===========================================================================
# 3. 通过 KEGG 基因 ID 查询对应通路 (批量 link/pathway)
# ===========================================================================
get_gene_pathways <- function(kegg_gene_ids) {
  message("\n[2] 查询 KEGG 基因→通路映射...")

  if (length(kegg_gene_ids) == 0) {
    return(data.frame(kegg_gene_id = character(), pathway_id = character(),
                      stringsAsFactors = FALSE))
  }

  all_results <- data.frame(
    kegg_gene_id = character(),
    pathway_id = character(),
    stringsAsFactors = FALSE
  )

  # 分批查询 (每批 10 个)
  batch_size <- 10
  n_batches <- ceiling(length(kegg_gene_ids) / batch_size)

  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(kegg_gene_ids))
    batch_ids <- kegg_gene_ids[start_idx:end_idx]
    batch_query <- paste(batch_ids, collapse = "+")

    cache_key <- paste0("gene_pathway_batch_", i)
    endpoint <- paste0("/link/pathway/", batch_query)
    lines <- kegg_rest(endpoint, cache_key = cache_key)

    if (length(lines) == 0) next

    for (line in lines) {
      parts <- strsplit(line, "\t")[[1]]
      if (length(parts) >= 2) {
        pathway_full <- sub("path:", "", parts[2])
        pathway_numeric <- gsub("^[a-z]+", "", pathway_full)
        all_results <- rbind(all_results, data.frame(
          kegg_gene_id = parts[1],
          pathway_id = pathway_full,
          pathway_code = pathway_numeric,
          stringsAsFactors = FALSE
        ))
      }
    }

    message(sprintf("    批次 %d/%d: %d 个基因, 累计 %d 通路链接",
                    i, n_batches, length(batch_ids), nrow(all_results)))
  }

  all_results
}

# ===========================================================================
# 4. 代谢物名称 → KEGG 化合物 ID 映射
# ===========================================================================
map_metabolites_to_kegg <- function(metabolite_names) {
  message("\n[3] 映射代谢物到 KEGG 化合物 ID...")

  results <- data.frame(
    metabolite_name = character(),
    kegg_compound_id = character(),
    kegg_description = character(),
    stringsAsFactors = FALSE
  )

  for (met in metabolite_names) {
    cache_key <- paste0("comp_find_", gsub("[^A-Za-z0-9]", "_", met))
    endpoint <- paste0("/find/compound/", URLencode(met, reserved = TRUE))
    lines <- kegg_rest(endpoint, cache_key = cache_key)

    if (length(lines) == 0 || all(lines == "")) {
      message(sprintf("    %s: 未找到 KEGG 化合物", met))
      next
    }

    parsed_lines <- strsplit(lines, "\t")
    best_match <- NULL

    for (pl in parsed_lines) {
      if (length(pl) < 2) next
      kegg_id <- pl[1]
      description <- pl[2]

      is_exact <- grepl(paste0("^", met, "$"), description, ignore.case = TRUE) ||
                  grepl(paste0("^", met, ";"), description, ignore.case = TRUE) ||
                  grepl(paste0("\\b", met, "\\b"), description, ignore.case = TRUE)

      if (is.null(best_match) || is_exact) {
        best_match <- list(kegg_id = kegg_id, description = description, exact = is_exact)
        if (is_exact) break
      }
    }

    if (!is.null(best_match)) {
      results <- rbind(results, data.frame(
        metabolite_name = met,
        kegg_compound_id = best_match$kegg_id,
        kegg_description = best_match$description,
        stringsAsFactors = FALSE
      ))
      message(sprintf("    %s -> %s",
                      met, best_match$kegg_id))
    } else {
      message(sprintf("    %s: 无最佳匹配", met))
    }
  }

  message(sprintf("\n  KEGG 化合物映射: %d/%d (%.1f%%)",
                  nrow(results), length(metabolite_names),
                  100 * nrow(results) / length(metabolite_names)))
  results
}

# ===========================================================================
# 5. 通过 KEGG 化合物 ID 查询对应通路
# ===========================================================================
get_compound_pathways <- function(kegg_compound_ids) {
  message("\n[4] 查询 KEGG 化合物→通路映射...")

  if (length(kegg_compound_ids) == 0) {
    return(data.frame(kegg_compound_id = character(), pathway_id = character(),
                      stringsAsFactors = FALSE))
  }

  all_results <- data.frame(
    kegg_compound_id = character(),
    pathway_id = character(),
    stringsAsFactors = FALSE
  )

  # 逐个查询 (化合物批量查询结果集很大, 单个查询更可控)
  for (i in seq_along(kegg_compound_ids)) {
    cid <- kegg_compound_ids[i]
    cache_key <- paste0("comp_pathway_", gsub("[:]", "_", cid))
    endpoint <- paste0("/link/pathway/", cid)
    lines <- kegg_rest(endpoint, cache_key = cache_key)

    if (length(lines) == 0) next

    for (line in lines) {
      parts <- strsplit(line, "\t")[[1]]
      if (length(parts) >= 2) {
        pathway_full <- sub("path:", "", parts[2])
        pathway_numeric <- gsub("^[a-z]+", "", pathway_full)
        all_results <- rbind(all_results, data.frame(
          kegg_compound_id = parts[1],
          pathway_id = pathway_full,
          pathway_code = pathway_numeric,
          stringsAsFactors = FALSE
        ))
      }
    }

    message(sprintf("    [%d/%d] %s: 累计 %d 通路链接",
                    i, length(kegg_compound_ids), cid, nrow(all_results)))
  }

  all_results
}

# ===========================================================================
# 6. 获取 KEGG 通路名称 (一次性下载所有小鼠通路列表)
# ===========================================================================
get_pathway_names <- function(pathway_ids = NULL) {
  message("\n[5] 查询 KEGG 通路名称 (一次性获取全部小鼠通路)...")

  # 一次性下载所有小鼠通路列表: list/pathway/mmu
  # 返回格式: "mmu00010  Glycolysis / Gluconeogenesis"
  lines <- kegg_rest("/list/pathway/mmu", cache_key = "all_mouse_pathways")

  if (length(lines) == 0) {
    message("  无法获取 KEGG 通路列表")
    return(data.frame(pathway_id = character(), pathway_name = character(),
                      pathway_code = character(),
                      stringsAsFactors = FALSE))
  }

  results <- data.frame(
    pathway_id = character(),
    pathway_name = character(),
    pathway_code = character(),
    stringsAsFactors = FALSE
  )

  for (line in lines) {
    parts <- strsplit(line, "\t")[[1]]
    if (length(parts) >= 2) {
      pathway_id <- parts[1]
      pathway_name <- parts[2]
      pathway_code <- gsub("^[a-z]+", "", pathway_id)

      results <- rbind(results, data.frame(
        pathway_id = pathway_id,
        pathway_name = pathway_name,
        pathway_code = pathway_code,
        stringsAsFactors = FALSE
      ))
    }
  }

  message(sprintf("  获取小鼠通路名称: %d 条", nrow(results)))

  if (!is.null(pathway_ids)) {
    pathway_codes_to_keep <- unique(gsub("^[a-z]+", "", pathway_ids))
    results <- results %>%
      filter(pathway_code %in% pathway_codes_to_keep)
    message(sprintf("  过滤到目标通路: %d 条", nrow(results)))
  }

  results
}

# ===========================================================================
# 7. 跨组学通路整合分析
# ===========================================================================
find_cross_omics_pathways <- function(gene_pathways, compound_pathways,
                                      gene_map, compound_map) {
  message("\n[6] 跨组学通路整合分析...")

  # 添加基因符号到通路映射
  gene_pathways_anno <- gene_pathways %>%
    left_join(gene_map %>% select(kegg_gene_id, human_symbol),
              by = "kegg_gene_id")

  compound_pathways_anno <- compound_pathways %>%
    left_join(compound_map %>% select(kegg_compound_id, metabolite_name),
              by = "kegg_compound_id")

  # 统计每个通路 (按 pathway_code) 有多少基因/代谢物映射
  gene_pathway_count <- gene_pathways_anno %>%
    filter(!is.na(pathway_code), pathway_code != "") %>%
    group_by(pathway_code) %>%
    summarise(
      n_genes = n_distinct(human_symbol),
      gene_list = paste(unique(human_symbol), collapse = ";"),
      .groups = "drop"
    )

  comp_pathway_count <- compound_pathways_anno %>%
    filter(!is.na(pathway_code), pathway_code != "") %>%
    group_by(pathway_code) %>%
    summarise(
      n_metabolites = n_distinct(metabolite_name),
      metabolite_list = paste(unique(metabolite_name), collapse = ";"),
      .groups = "drop"
    )

  # 找出共享通路 (使用 pathway_code 匹配, 因为 mmu/map 前缀不同)
  shared_pathways <- inner_join(
    gene_pathway_count, comp_pathway_count,
    by = "pathway_code"
  ) %>%
    mutate(
      cross_omics_score = n_genes + n_metabolites,
      total_omics_elements = n_genes + n_metabolites
    ) %>%
    arrange(desc(cross_omics_score))

  message(sprintf("  共享 KEGG 通路: %d 个", nrow(shared_pathways)))

  list(
    gene_pathway_count = gene_pathway_count,
    comp_pathway_count = comp_pathway_count,
    shared_pathways = shared_pathways,
    gene_pathways_anno = gene_pathways_anno,
    compound_pathways_anno = compound_pathways_anno
  )
}

# ===========================================================================
# 8. 可视化: 跨组学通路气泡图
# ===========================================================================
plot_cross_omics_pathways <- function(shared_pathways) {
  if (nrow(shared_pathways) == 0) {
    message("  无共享通路, 跳过可视化")
    return(invisible(NULL))
  }

  plot_data <- shared_pathways %>%
    head(25) %>%
    mutate(
      pathway_label = ifelse(is.na(pathway_name),
                              paste0("KEGG:", pathway_code), pathway_name),
      pathway_label = substr(pathway_label, 1, 65)
    ) %>%
    arrange(cross_omics_score)

  plot_data$pathway_label <- factor(plot_data$pathway_label,
                                    levels = plot_data$pathway_label)

  p <- ggplot(plot_data, aes(
    x = n_genes,
    y = n_metabolites,
    size = cross_omics_score,
    color = cross_omics_score,
    label = pathway_label
  )) +
    geom_point(alpha = 0.7) +
    ggrepel::geom_text_repel(
      size = 3,
      max.overlaps = 25,
      box.padding = 0.4
    ) +
    scale_color_gradient(low = "#FDAE61", high = "#B2182B",
                         name = "Cross-omics Score") +
    scale_size_continuous(name = "Cross-omics Score", range = c(3, 12)) +
    labs(
      title = "跨组学共享 KEGG 通路",
      subtitle = "X轴: FA-96 基因数 | Y轴: 跨数据集一致代谢物数",
      x = "FA-96 基因数",
      y = "一致代谢物数"
    ) +
    theme_bw(base_size = 11)

  ggsave(file.path(FIG_DIR, "cross_omics_kegg_pathways.pdf"),
         p, width = 13, height = 9, dpi = 300)
  message("跨组学通路气泡图已保存")

  # 第二张图: 条形图 (按 pathway_id 分组显示基因+代谢物)
  bar_data <- plot_data %>%
    select(pathway_label, n_genes, n_metabolites) %>%
    pivot_longer(cols = c(n_genes, n_metabolites),
                 names_to = "omics_type", values_to = "count") %>%
    mutate(omics_type = recode(omics_type,
                                n_genes = "FA-96 基因",
                                n_metabolites = "一致代谢物"))

  p2 <- ggplot(bar_data, aes(
    x = pathway_label,
    y = count,
    fill = omics_type
  )) +
    geom_col(position = "dodge", color = "black", linewidth = 0.3) +
    coord_flip() +
    scale_fill_manual(values = c("#1F78B4", "#E31A1C"),
                      name = "组学类型") +
    labs(
      title = "跨组学共享 KEGG 通路: 基因 vs 代谢物数量",
      x = "KEGG 通路",
      y = "分子数"
    ) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom")

  ggsave(file.path(FIG_DIR, "cross_omics_kegg_barplot.pdf"),
         p2, width = 12, height = 9, dpi = 300)
  message("跨组学通路条形图已保存")
}

# ===========================================================================
# 9. 主流程
# ===========================================================================
main <- function() {
  message("========================================")
  message("KEGG 通路水平跨组学整合")
  message("========================================")

  # 9.1 加载 FA-96 基因集
  message("\n[A] 加载 FA-96 基因集...")
  if (!file.exists(FA_GENE_CSV)) {
    stop("FA-96 基因集文件不存在: ", FA_GENE_CSV)
  }
  fa_genes_df <- read.csv(FA_GENE_CSV, stringsAsFactors = FALSE)
  fa_genes <- fa_genes_df$gene_symbol
  message(sprintf("  FA-96 基因数: %d", length(fa_genes)))

  # 9.2 加载跨数据集一致代谢物
  message("\n[B] 加载跨数据集一致代谢物...")
  if (!file.exists(CROSS_META_CSV)) {
    stop("跨数据集一致代谢物文件不存在: ", CROSS_META_CSV,
         "\n请先运行 15_cross_validation_mcao.R")
  }
  cross_mets <- read.csv(CROSS_META_CSV, stringsAsFactors = FALSE)
  message(sprintf("  跨数据集一致代谢物: %d", nrow(cross_mets)))

  # metabolite names for KEGG search - 用更通用的搜索词
  met_search_terms <- unique(c(
    cross_mets$display_name,
    "4-hydroxynonenal", "malondialdehyde",
    "12-HETE", "15-HETE", "arachidonic acid",
    "spermidine", "spermine", "putrescine", "ornithine",
    "glutathione", "cysteine", "glutamate", "taurine",
    "citrate", "succinate", "fumarate", "malate",
    "NAD", "nicotinamide", "lactate", "pyruvate",
    "ceramide", "sphingosine", "choline", "ethanolamine",
    "palmitic acid", "stearic acid", "oleic acid",
    "LPC", "phosphatidylethanolamine"
  ))
  message(sprintf(" 代谢物搜索词数: %d", length(met_search_terms)))

  # 9.3 映射 FA-96 基因到 KEGG
  message("\n[C] FA-96 基因 → KEGG 小鼠同源...")
  gene_map <- map_genes_to_kegg(fa_genes)
  write.csv(gene_map, file.path(TAB_DIR, "fa96_gene_kegg_mapping.csv"),
            row.names = FALSE)

  # 9.4 查询基因通路
  message("\n[D] FA-96 基因 → KEGG 通路...")
  gene_pathways <- get_gene_pathways(gene_map$kegg_gene_id)
  write.csv(gene_pathways,
            file.path(TAB_DIR, "fa96_gene_pathway_links.csv"),
            row.names = FALSE)
  message(sprintf("  基因-通路链接: %d", nrow(gene_pathways)))

  # 9.5 映射代谢物到 KEGG 化合物
  message("\n[E] 代谢物 → KEGG 化合物 ID...")
  compound_map <- map_metabolites_to_kegg(met_search_terms)
  write.csv(compound_map,
            file.path(TAB_DIR, "metabolite_kegg_compound_mapping.csv"),
            row.names = FALSE)

  # 9.6 查询化合物通路
  message("\n[F] 化合物 → KEGG 通路...")
  compound_pathways <- get_compound_pathways(compound_map$kegg_compound_id)
  write.csv(compound_pathways,
            file.path(TAB_DIR, "metabolite_pathway_links.csv"),
            row.names = FALSE)
  message(sprintf("  化合物-通路链接: %d", nrow(compound_pathways)))

  # 9.7 跨组学通路整合
  message("\n[G] 跨组学通路整合...")
  cross_omics <- find_cross_omics_pathways(
    gene_pathways, compound_pathways, gene_map, compound_map
  )

  # 9.8 获取通路名称 (一次性获取所有小鼠通路)
  message("\n[H] 获取 KEGG 通路名称...")
  pathway_names <- get_pathway_names()

  # 9.9 输出表格
  message("\n[I] 保存结果...")

  shared_final <- cross_omics$shared_pathways %>%
    left_join(pathway_names %>% select(pathway_code, pathway_name),
              by = "pathway_code") %>%
    arrange(desc(cross_omics_score))

  write.csv(shared_final,
            file.path(TAB_DIR, "cross_omics_shared_pathways.csv"),
            row.names = FALSE)

  write.csv(cross_omics$gene_pathway_count,
            file.path(TAB_DIR, "fa96_gene_pathway_count.csv"),
            row.names = FALSE)

  write.csv(cross_omics$comp_pathway_count,
            file.path(TAB_DIR, "metabolite_pathway_count.csv"),
            row.names = FALSE)

  message(sprintf("\n  跨组学共享 KEGG 通路: %d 个", nrow(shared_final)))

  # 9.10 可视化
  message("\n[J] 生成可视化...")
  plot_cross_omics_pathways(shared_final)

  # 9.11 总结
  message("\n========================================")
  message("KEGG 通路水平跨组学整合完成!")
  message(sprintf("  输出目录: %s", OUT_DIR))
  message(sprintf("  图表: %s", FIG_DIR))
  message(sprintf("  表格: %s", TAB_DIR))
  message("========================================")

  if (nrow(shared_final) > 0) {
    message("\nTop 10 跨组学共享 KEGG 通路:")
    top_paths <- head(shared_final, 10)
    for (i in seq_len(nrow(top_paths))) {
      p <- top_paths[i, ]
      pname <- ifelse(is.na(p$pathway_name), p$pathway_id, p$pathway_name)
      message(sprintf("  [%d] %s | 基因:%d | 代谢物:%d | 分数:%d",
                      i, substr(pname, 1, 60),
                      p$n_genes, p$n_metabolites, p$cross_omics_score))
    }
  }

  results <- list(
    gene_map = gene_map,
    compound_map = compound_map,
    gene_pathways = gene_pathways,
    compound_pathways = compound_pathways,
    cross_omics = cross_omics,
    shared_pathways = shared_final,
    pathway_names = pathway_names
  )

  saveRDS(results, file.path(OUT_DIR, "kegg_pathway_integration_results.rds"))

  invisible(results)
}

# ===========================================================================
# 10. 执行
# ===========================================================================
results <- main()
