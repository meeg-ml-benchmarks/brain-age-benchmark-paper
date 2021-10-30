library(ggplot2)
library(scales)
library(ggthemes)
library(patchwork)
source('utils.r')

# fig 1

results_fnames <- list.files('results/', full.names = T)

results <- do.call(rbind, lapply(results_fnames, read.csv))
results$X <- NULL

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
  facet_wrap(dataset~., nrow = 4, strip.position="right") +
  geom_jitter(show.legend = F, size = 1.2, alpha = 0.7) +
  geom_boxplot(alpha=0.1, show.legend = F, size = 1.1, width = 0.8) +
  theme_minimal(base_size = 16) +
  coord_cartesian(xlim=c(-0.3, 0.8)) +
  scale_x_continuous(breaks = seq(-0.5, 0.8, 0.1)) +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values) +
  geom_vline(xintercept=0, color = 'black', linetype='dashed') +
  labs(y = element_blank(), x = bquote(R^2~"[10-fold cross validation]"),
       title = 'Age Prediction From M/EEG Signals'))

my_ggsave('./figures/fig_performance_r2', fig_r2, dpi = 300, width = 8, height = 6)


set.seed(42)
(fig_mae <- ggplot(
  aes(x = MAE, y = benchmark, color = benchmark,
      fill = benchmark),
      data = results) +
  facet_wrap(dataset~., nrow = 4, strip.position="right") +
  geom_jitter(show.legend = F, size = 1.2, alpha = 0.7) +
  geom_boxplot(alpha=0.1, show.legend = F, size = 1.1, width = 0.8) +
  theme_minimal(base_size = 16) +
  scale_x_continuous(breaks = seq(0, 25, 1)) +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values) +
  labs(y = element_blank(), x = "MAE [10-fold cross validation]",
      title = 'Age Prediction From M/EEG Signals'))

my_ggsave('./figures/fig_performance_mae', fig_mae, dpi = 300,  width = 8, height = 6)
