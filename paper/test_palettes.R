# Verify project palettes load correctly and render a preview figure

suppressPackageStartupMessages({
  library(ggplot2)
})

source("d:/铁衰老 绝不重蹈覆辙/paper/palettes.R")

outdir <- "d:/铁衰老 绝不重蹈覆辙/paper/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

pals <- load_palettes()
names <- setdiff(names(pals), "_meta")

cat("Available palettes:\n")
cat(paste(" ", names, collapse = "\n"), "\n")

# Build a long data frame for ggplot2
df_list <- lapply(names, function(nm) {
  cols <- get_palette(nm)
  data.frame(
    palette = nm,
    idx = seq_along(cols),
    color = cols,
    stringsAsFactors = FALSE
  )
})
df <- do.call(rbind, df_list)
df$palette <- factor(df$palette, levels = names)

p <- ggplot(df, aes(x = idx, y = palette, fill = color)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_identity() +
  labs(title = "Project Publication Palette Preview",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.text.x = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )

png_path <- file.path(outdir, "palette_preview_r.png")
pdf_path <- file.path(outdir, "palette_preview_r.pdf")

ggsave(png_path, p, width = 10, height = 0.6 * length(names) + 1, dpi = 300, units = "in")
ggsave(pdf_path, p, width = 10, height = 0.6 * length(names) + 1, units = "in")

cat(sprintf("[OK] PNG: %s\n", png_path))
cat(sprintf("[OK] PDF: %s\n", pdf_path))
cat("[DONE] All palettes verified.\n")
