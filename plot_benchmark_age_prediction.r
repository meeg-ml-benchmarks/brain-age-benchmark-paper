library(ggplot2)
library(scales)
library(ggthemes)   
# fig 1

results_fnames <- list.files('results/', full.names = T)

results <- do.call(rbind, lapply(results_fnames, read.csv))
results$X <- NULL

# order dataset/benchmark by average scores
data_agg <- aggregate(
  score ~ dataset,
  subset(results, metric == 'r2' & benchmark != 'dummy'),
  FUN = mean)

data_order <- data_agg$dataset[order(data_agg$score)]
results$dataset <- factor(results$dataset, levels = data_order)

bench_agg <- aggregate(
  score ~ benchmark,
  subset(results, metric == 'r2'),
  FUN = mean)

bench_order <- bench_agg$benchmark[order(bench_agg$score)]
results$benchmark <- factor(results$benchmark, levels = rev(bench_order))

ggplot(
  aes(x = score, y = benchmark, color = benchmark,
      fill = benchmark,
      shape = dataset) ,
  data = subset(results, metric == 'r2')) +
  facet_wrap(dataset~., nrow = 4, strip.position="right") +
  geom_jitter(show.legend = F) +
  geom_boxplot(alpha=0.1, show.legend = F) +
  theme_minimal(base_size = 18) +
  scale_color_colorblind() +
  scale_fill_colorblind() +
  labs(y = element_blank(), x = "age prediction R2 [10-fold CV]")


ggplot(
  aes(x = score, y = benchmark, color = benchmark,
      fill = benchmark,
      shape = dataset) ,
  data = subset(results, metric == 'r2')) +
  facet_wrap(dataset~., nrow = 4, strip.position="right") +
  geom_jitter(show.legend = F) +
  geom_boxplot(alpha=0.1, show.legend = F) +
  theme_minimal(base_size = 18) +
  scale_color_colorblind() +
  scale_fill_colorblind() +
  labs(y = element_blank(), x = "age prediction MAE [10-fold CV]")
