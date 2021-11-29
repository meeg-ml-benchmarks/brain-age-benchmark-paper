library(ggplot2)
library(scales)
library(ggthemes)
library(patchwork)
library(kableExtra)
source('utils.r')

# fig 1

results_fnames <- list.files('results/', full.names = T)

results <- do.call(rbind, lapply(results_fnames, read.csv))
results$X <- NULL

results$dataset <- factor(
  results$dataset,
  levels = c('camcan', 'lemon', 'chbp', 'tuab'),
  labels = c('Cam-CAN', 'LEMON', 'CHBP', 'TUAB'))

colors <- setNames(
  colorblind_pal()(8),
  c('black', 'orange', 'skye_blue', 'bluish_green', 'yellow', 'blue',
    'vermillon', 'purple'))


results$benchmark <- factor(
  results$benchmark,
  levels = c('deep', 'shallow', 'filterbank-source', 'filterbank-riemann', 'handcrafted','dummy'))


color_values <- as.vector(colors[c('vermillon', 'orange', 'blue',
                                   'skye_blue', 'bluish_green', 'black')])

set.seed(42)
(fig_r2 <- ggplot(
  aes(x = r2, y = benchmark, color = benchmark,
      fill = benchmark),
      data = results) +
  facet_wrap(dataset~., nrow = 4, strip.position = "right") +
  geom_jitter(show.legend = F, size = 1.2, alpha = 0.7) +
  geom_boxplot(alpha = 0.1, show.legend = F, size = 1.1, width = 0.8) +
  theme_minimal(base_size = 16) +
  coord_cartesian(xlim = c(-0.3, 0.8)) +
  scale_x_continuous(breaks = seq(-0.5, 0.8, 0.1)) +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values) +
  geom_vline(xintercept = 0, color = 'black', linetype='dashed') +
  labs(y = element_blank(), x = bquote(R^2~"[10-fold cross validation]"),
       title = 'Age Prediction From M/EEG Signals'))

my_ggsave('./figures/fig_performance_r2', fig_r2, dpi = 300, width = 8, height = 6)


set.seed(42)
(fig_mae <- ggplot(
  aes(x = MAE, y = benchmark, color = benchmark,
      fill = benchmark),
      data = results) +
  facet_wrap(dataset~., nrow = 4, strip.position = "right") +
  geom_jitter(show.legend = F, size = 1.2, alpha = 0.7) +
  geom_boxplot(alpha = 0.1, show.legend = F, size = 1.1, width = 0.8) +
  theme_minimal(base_size = 16) +
  scale_x_continuous(breaks = seq(0, 25, 1)) +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values) +
  labs(y = element_blank(), x = "MAE [10-fold cross validation]",
      title = 'Age Prediction From M/EEG Signals'))

my_ggsave('./figures/fig_performance_mae', fig_mae, dpi = 300,  width = 8, height = 6)


# Make table
results_out <- results[,c('MAE', 'r2', 'benchmark', 'dataset')]

agg_cv <- aggregate(r2 ~ benchmark + dataset, data = results_out, FUN = mean)
agg_cv$r2sd <- aggregate(r2 ~ benchmark + dataset, data = results_out, FUN = sd)$r2
names(agg_cv)[3:4] <- c(expression(R^2), "+/-")
agg_cv$MAE <- aggregate(MAE ~ benchmark + dataset, data = results_out, FUN = mean)$MAE
agg_cv$MAEsd <- aggregate(MAE ~ benchmark + dataset, data = results_out, FUN = sd)$MAE
names(agg_cv)[6] <- "+/-"
agg_cv <- agg_cv[,c('dataset', 'benchmark', names(agg_cv)[-c(1, 2)])]
for (ii in 3:6)
{
  cell_spec(agg_cv[, ii], format = 'r')
}
tab <- kbl(agg_cv, caption = "Table 1. Cross-validation results across benchmarks and datasets", digits = 2)
tab <- kable_classic(tab, full_width = F, html_font = "Arial")
tab <- column_spec(tab, 1:2, width = "10em")
tab <- column_spec(tab, 3:6, width = "5em")
save_kable(tab, 'tables/cv_table.html', self_contained = T)
