# Project-wide color palettes for publication-quality figures
# Compatible with ggplot2 via scale_*_manual()

library(jsonlite)

#' Load all palettes from palettes.json
#'
#' @param path Path to palettes.json. Defaults to "paper/palettes.json" relative
#'   to the project root or to the directory of the calling script.
#' @return A named list of palettes.
load_palettes <- function(path = NULL) {
  if (is.null(path)) {
    path <- "d:/铁衰老 绝不重蹈覆辙/paper/palettes.json"
  }
  if (!file.exists(path)) {
    stop("Palette file not found: ", path)
  }
  fromJSON(path, simplifyDataFrame = FALSE)
}

#' Get a single palette vector
#'
#' @param name Palette name, e.g. "nature_cancer", "nature_npg", "okabe_ito".
#' @param n Optional number of colors to return (recycled if needed).
#' @param path Path to palettes.json.
#' @return Character vector of hex colors.
get_palette <- function(name, n = NULL, path = NULL) {
  pals <- load_palettes(path)
  if (!name %in% names(pals)) {
    stop("Unknown palette: ", name, ". Available: ", paste(names(pals), collapse = ", "))
  }
  colors <- pals[[name]]$colors
  if (!is.null(n)) {
    colors <- rep(colors, length.out = n)
  }
  colors
}

#' ggplot2 scale wrapper for project palettes
#'
#' @param name Palette name.
#' @param ... Additional arguments passed to scale_colour_manual / scale_fill_manual.
#' @param aesthetics Which aesthetics to apply: "colour", "fill", or both.
#' @return ggplot2 scale object.
scale_color_project <- function(name, ..., aesthetics = "colour") {
  cols <- get_palette(name)
  if ("colour" %in% aesthetics || "color" %in% aesthetics) {
    return(ggplot2::scale_colour_manual(values = cols, ...))
  }
  ggplot2::scale_color_manual(values = cols, ...)
}

scale_fill_project <- function(name, ...) {
  cols <- get_palette(name)
  ggplot2::scale_fill_manual(values = cols, ...)
}

#' Convenience: Nature Cancer palette
nature_cancer_pal <- function(n = NULL) get_palette("nature_cancer", n = n)
nature_npg_pal    <- function(n = NULL) get_palette("nature_npg", n = n)
okabe_ito_pal     <- function(n = NULL) get_palette("okabe_ito", n = n)
