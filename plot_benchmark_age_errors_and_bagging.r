library(ggplot2)
library(scales)
library(ggthemes)
library(patchwork)
library(kableExtra)
library(data.table)
source('utils.r')

# fig 1

results_fnames <- list.files('results/', full.names = T)

results_fnames <- results_fnames[grepl('ys.csv', results_fnames, fixed = T)]

results_dummy <- results_fnames[grepl('dummy', results_fnames, fixed = T)]

results_fnames <- results_fnames[!grepl('dummy', results_fnames, fixed = T)]


read_data_and_get_info <- function(x)
{

    dat <- read.csv(x)
    name <- strsplit(x[[1]], '/')[[1]][3]
    name <- gsub('filterbank-', 'filterbank/', name) 
    within(dat,
    {
        benchmark = strsplit(
            strsplit(name, '_')[[1]][1], '-')[[1]][2]
        dataset = strsplit(
            strsplit(name, '_')[[1]][2], '-')[[1]][2]
    })
}

read_data_and_get_info2 <- function(x)
{
    out <- read_data_and_get_info(x)
    out
}

results_ <- do.call(rbind, lapply(results_dummy, read_data_and_get_info))
results_$X <- NULL
# XXX unless fixed, remove split info

results <- do.call(rbind, lapply(results_fnames, read_data_and_get_info2))
results$X <- NULL

results <- rbind(results_, results)

results$dataset <- factor(
  results$dataset,
  levels = c('camcan', 'lemon', 'chbp', 'tuab'),
  labels = c('Cam-CAN\n(MEG)', 'LEMON\n(EEG)', 'CHBP\n(EEG)', 'TUAB\n(EEG)'))

colors <- setNames(
  colorblind_pal()(8),
  c('black', 'orange', 'skye_blue', 'bluish_green', 'yellow', 'blue',
    'vermillon', 'purple'))

results$benchmark <- factor(
  results$benchmark,
  levels = c('deep', 'shallow', 'filterbank/source', 'filterbank/riemann',
             'handcrafted','dummy'),
  labels = c('deep', 'shallow', 'filterbank\nsource', 'filterbank\nriemann',
             'handcrafted','dummy'))

color_values <- as.vector(colors[c('vermillon', 'orange', 'blue',
                                   'skye_blue', 'bluish_green', 'black')])

agg_cv <- read.csv('./results_agg_cv.csv')
agg_cv$benchmark <- factor(
  agg_cv$benchmark,
  levels = c('deep', 'shallow', 'filterbank-source', 'filterbank-riemann',
             'handcrafted','dummy'),
  labels = c('deep', 'shallow', 'filterbank\nsource', 'filterbank\nriemann',
             'handcrafted','dummy'))

agg_cv$label <- paste0("R^2%==%", round(agg_cv$r2, 2))

fig_scat1 <- ggplot(data = subset(results, benchmark != 'dummy'), mapping = aes(
    x = y_true, y = y_pred, color = benchmark, fill = benchmark
)) +
    geom_point(show.legend = F, size = 0.3) +
    facet_grid(dataset ~ benchmark) +
    coord_cartesian(ylim = c(0, 100), xlim = c(0, 100)) +
    theme_minimal(base_size = 16) +
    scale_color_manual(values = color_values) +
    geom_abline(intercept = 0, slope = 1, size = 0.2) +
    labs(x = 'Age [years]', y = 'Brain Age [years]') +
    geom_text(data = subset(agg_cv, benchmark != 'dummy'),
              mapping = aes(label = label), x = 25, y = 90,
              size = 3,
              inherit.aes = F, parse = T)

my_ggsave('./figures/fig_performance_scatter', plot = fig_scat1, dpi = 300, 
          width = 8, height = 7)

dt_results <- data.table(results)

dt_results <- dt_results[benchmark != 'dummy']
dt_results$benchmark <- factor(dt_results$benchmark)
split_list <- split(dt_results, dt_results$benchmark)
dt_ba <- do.call(cbind, lapply(split_list, function(dt) {
    dt$y_pred
}))

dt_ba <- data.table(dt_ba)
dt_ba$dataset <- split_list[[1]]$dataset
dt_ba$model_mean1 <- rowMeans(dt_ba[, names(split_list), with = F])
dt_ba$model_sd1 <-  apply(dt_ba[, names(split_list), with = F], 1, sd)
dt_ba$model_mean2 <- rowMeans(dt_ba[, names(split_list)[-5], with = F])
dt_ba$model_sd2 <-  apply(dt_ba[, names(split_list)[-5], with = F], 1, sd)

ba_loa <- rbindlist(lapply(
    split(dt_ba, dt_ba$dataset),
    function(x)
    {
        loa1 <- qchisq(.95, df = 4) * (1 / sqrt(4)) * mean(x$model_sd1)
        loa2 <- qchisq(.95, df = 3) * (1 / sqrt(3)) * mean(x$model_sd2)
        data.table(loa1 = loa1, loa2 = loa2, dataset = x$dataset)
    }
))

ba_plot <- ggplot(data = dt_ba, mapping = aes(x = model_mean1, y = model_sd1, color = dataset)) +
    geom_point(show.legend = F, size = 1) +
    facet_wrap(~dataset, scales = 'free_y') +
    theme_minimal(base_size = 16) +
    scale_color_brewer(palette = 'Dark2') +
    geom_hline(data = ba_loa, mapping = aes(yintercept = loa1), linetype = 'dashed') +
    labs(x = 'mean of age prediction', y = 'standard deviation of age prediction',
        title = '(B) Multirater Bland-Altman Plot')

dt_results_bagged <-
    dt_results[benchmark != 'dummy', .(y_pred = mean(y_pred), y_true = unique(y_true)),
    .(cv_split,dataset,subject)] 

dt_scores_bagged <- dt_results_bagged[
    , .(MAE = mean(abs(y_true - y_pred)),
        R2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)),
    .(cv_split, dataset)]


agg_cv2 <- read.csv('./results_agg_cv2.csv')
agg_cv2$benchmark <- factor(
  agg_cv2$benchmark,
  levels = c('deep', 'shallow', 'filterbank-source', 'filterbank-riemann',
             'handcrafted','dummy'),
  labels = c('deep', 'shallow', 'filterbank\nsource', 'filterbank\nriemann',
             'handcrafted','dummy'))

best_mods <- rbindlist(lapply(split(agg_cv2, agg_cv2$dataset), function(x) {
    idx <- which.max(x[['r2']])
    best_score <- x[['r2']][idx]
    best_model <- x[['benchmark']][idx]
    data.table(score = best_score, model = best_model, dataset = x$dataset)
}))

bag_plot <- ggplot(data = dt_scores_bagged, mapping = aes(x = R2, y = dataset, color = dataset, fill = dataset)) +
  geom_jitter(show.legend = F, size = 1.2, alpha = 0.7) +
  geom_boxplot(alpha = 0.1, show.legend = F, size = 1.1, width = 0.8,
               outlier.alpha = 0) +
  theme_minimal(base_size = 16) +
  scale_color_brewer(palette = 'Dark2') +
  scale_fill_brewer(palette = 'Dark2') +
  coord_cartesian(xlim = c(-0.3, 0.8)) +
  scale_x_continuous(breaks = seq(-0.5, 0.8, 0.1)) +
  geom_vline(xintercept = 0, color = 'black', linetype='dashed') +
  geom_point(data = best_mods,
             mapping = aes(x = score, y = dataset), inherit.aes = F,
             size = 4, shape = 18) +
  labs(y = element_blank(), x = bquote(R^2~"[10-fold cross validation]"),
       title = '(B) Bagged Performance')

fig_ba_plot_final <- ba_plot | bag_plot

my_ggsave('./figures/fig_ba_plot_bagging', plot = fig_ba_plot_final, dpi = 300, 
          width = 10, height = 5)