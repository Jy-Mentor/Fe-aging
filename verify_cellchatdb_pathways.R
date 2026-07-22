##############################################################################
# 加载本地 CellChatDB.mouse.rda, 提取真实通路列表, 验证 28 个候选通路
##############################################################################
load("d:/铁衰老 绝不重蹈覆辙/L2/results/CellChatDB.mouse.rda")

cat("===== CellChatDB.mouse structure =====\n")
cat("Class:", class(CellChatDB.mouse), "\n")
cat("Names:", paste(names(CellChatDB.mouse), collapse=", "), "\n\n")

inter <- CellChatDB.mouse$interaction
cat("Interaction dim:", paste(dim(inter), collapse=" x "), "\n")
cat("Cols:", paste(colnames(inter), collapse=", "), "\n\n")

# pathway_name 提取
pw_col <- inter$pathway_name
cat("pathway_name class:", class(pw_col), "\n")
pw_vec <- as.character(pw_col)
pw_unique <- sort(unique(pw_vec))
cat("\n===== CellChatDB.mouse pathway list (total:", length(pw_unique), ") =====\n")
cat(paste(pw_unique, collapse="\n"), "\n")

# annotation 统计
cat("\n===== Annotation distribution =====\n")
ann_col <- as.character(inter$annotation)
print(table(ann_col, useNA="ifany"))

# 28 个候选通路核对
cat("\n===== 28 candidate pathways check =====\n")
cand <- c("SPP1","TGFb","CXCL","CCL","TNF","IL6","GALECTIN","MIF","COMPLEMENT",
          "FLT3","GRN","VISFATIN","NRXN","NCAM","NOTCH","WNT","BMP","FGF","VEGF",
          "PDGF","EGF","IL1","IL2","IL4","IL10","IL12","IL16","IL17")
in_db <- character(0)
not_in_db <- character(0)
for (p in cand) {
  hit <- p %in% pw_unique
  cat(sprintf("  %-12s : %s\n", p, ifelse(hit, "IN DB", "*** NOT IN DB ***")))
  if (hit) in_db <- c(in_db, p) else not_in_db <- c(not_in_db, p)
}
cat("\n--- Summary ---\n")
cat("In DB      :", length(in_db), ":", paste(in_db, collapse=", "), "\n")
cat("NOT in DB  :", length(not_in_db), ":", paste(not_in_db, collapse=", "), "\n")

# 对每个候选通路, 列出其 L-R 对数量
cat("\n===== L-R pair counts per candidate pathway =====\n")
for (p in cand) {
  n <- sum(pw_vec == p)
  if (n > 0) {
    ex <- inter[inter$pathway_name == p, c("interaction_name","ligand","receptor")]
    cat(sprintf("\n[%s]  %d L-R pairs\n", p, n))
    print(head(ex, 5))
  }
}

# 写出完整通路表
out <- data.frame(pathway_name = pw_vec, annotation = ann_col,
                  stringsAsFactors = FALSE)
out_unique <- unique(out)
out_unique <- out_unique[order(out_unique$pathway_name), ]
write.csv(out_unique,
          "d:/铁衰老 绝不重蹈覆辙/L2/results/cellchatdb_mouse_pathways.csv",
          row.names = FALSE)
cat("\nSaved: cellchatdb_mouse_pathways.csv\n")