library(ggplot2)
library(scales)
library(ggthemes)
library(patchwork)
source('utils.r')

# fig 1

results_fnames <- list.files('results/', full.names = T)

results <- do.call(rbind, lapply(results_fnames, read.csv))
results$X <- NULL

# order dataset/benchmark by average scores
data_agg <- aggregate(
  r2 ~ dataset,
  subset(results, benchmark != 'dummy'),
  FUN = mean)

data_order <- data_agg$dataset[order(data_agg$r2)]
results$dataset <- factor(results$dataset, levels = data_order)

bench_agg <- aggregate(
  r2 ~ benchmark,
  results,
  FUN = mean)

bench_order <- bench_agg$benchmark[order(bench_agg$r2)]
results$benchmark <- factor(results$benchmark, levels = rev(bench_order))

colors <- setNames(
  colorblind_pal()(8),
  c('black', 'orange', 'skye_blue', 'bluish_green', 'yellow', 'blue',
    'vermillon', 'purple'))

color_values <- as.vector(colors[c('bluish_green', 'orange', 'skye_blue',
                                   'black', 'vermillon', 'purple')])

set.seed(42)
(fig_r2 <- ggplot(
  aes(x = r2, y = benchmark, color = benchmark,
      fill = benchmark),
      data = results) +
  facet_wrap(dataset~., nrow = 4, strip.position="right") +
  geom_jitter(show.legend = F, size = 1.7, alpha = 0.7) +
  geom_boxplot(alpha=0.1, show.legend = F, size = 1.2, width = 0.9) +
  theme_minimal(base_size = 24) +
  coord_cartesian(xlim=c(-0.5, 1)) +
  scale_y_discrete(labels = element_blank()) +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values) +
  geom_vline(xintercept=0, color = 'black', linetype='dashed') +
  labs(y = element_blank(), x = expression(R^2), title = 'B')) +
  theme(plot.title = element_text(face = 'bold'))

# my_ggsave('./figures/fig_performance_r2', fig_r2, dpi = 300, width = 10, height = 8)


set.seed(42)
(fig_mae <- ggplot(
  aes(x = MAE, y = benchmark, color = benchmark,
      fill = benchmark),
      data = results) +
  facet_wrap(dataset~., nrow = 4, strip.position="right") +
  geom_jitter(show.legend = F, size = 1.7, alpha = 0.7) +
  geom_boxplot(alpha=0.1, show.legend = F, size = 1.2, width = 0.9) +
  theme_minimal(base_size = 24) +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values) +
  labs(y = element_blank(), x = "MAE", title = 'A')) +
  theme(plot.title = element_text(face = 'bold'))


p <- fig_mae | fig_r2
p + plot_annotation(
  title = 'Age Prediction From M/EEG [10-fold cross validation]',
  theme = theme(plot.title = element_text(size = 26))
)

my_ggsave('./figures/fig_performance', p, dpi = 300,
          width = 12, height = 7.5)


set.seed(42)
(fig_fit <- ggplot(
  aes(x = fit_time, y = benchmark, color = benchmark,
      fill = benchmark),
      data = results) +
  facet_wrap(dataset~., nrow = 4, strip.position="right") +
  geom_jitter(show.legend = F, size = 1.7, alpha = 0.7) +
  geom_boxplot(alpha=0.1, show.legend = F, size = 1.2, width = 0.9) +
  theme_minimal(base_size = 24) +
  scale_x_log10() +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values) +
  labs(y = element_blank(), x = "fit time [seconds]",
       title = 'A')) +
  theme(plot.title = element_text(face = 'bold'))

set.seed(42)
(fig_score <- ggplot(
  aes(x = score_time, y = benchmark, color = benchmark,
      fill = benchmark),
      data = results) +
  facet_wrap(dataset~., nrow = 4, strip.position="right") +
  geom_jitter(show.legend = F, size = 1.7, alpha = 0.7) +
  geom_boxplot(alpha=0.1, show.legend = F, size = 1.2, width = 0.9) +
  theme_minimal(base_size = 24) +
  scale_x_log10() +
  scale_y_discrete(labels = element_blank()) +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values) +
  labs(y = element_blank(), x = "score time [seconds]",
       title = 'B')) +
  theme(plot.title = element_text(face = 'bold'))

p2 <- fig_fit | fig_score
p2 + plot_annotation(
  title = 'Run time [10-fold cross validation]',
  theme = theme(plot.title = element_text(size = 26))
)

my_ggsave('./figures/fig_runtime', p2, dpi = 300,
          width = 12, height = 7.5)
# my_ggsave('./figures/fig_performance_mae', fig, dpi = 300, width = 10, height = 8)
