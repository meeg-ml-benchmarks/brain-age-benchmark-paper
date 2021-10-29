
my_ggsave <- function(fname, plot, dpi = 300, ...)
{
  ggsave(paste0(fname, ".pdf"), plot = plot, useDingbats = F, bg = "#FFFFFF",
         ...)
  embedFonts(file = paste0(fname, ".pdf"),
             outfile = paste0(fname, ".pdf"))
  ggsave(paste0(fname, ".png"), plot = plot, dpi = dpi, bg = "#FFFFFF",
         ...)
}