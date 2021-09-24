library(ggplot2)
library(scales)     
# fig 1

set.seed(42)
demog_data_camcan <- data.frame(
    age = c(rgamma(300, 90, 2), rgamma(320, 100, 2)),
    gender = c(rep("F", 300), rep("M", 320)),
    dataset = "cam-can"
)
demog_data_tuab <- data.frame(
    age = c(rgamma(700, 100, 2), rgamma(639, 95, 2)),
    gender = c(rep("F", 700), rep("M", 639)),
    dataset = "tuab"
)
demog_data_lemon <- data.frame(
    age = c(rgamma(90, 105, 2), rgamma(100, 100, 2)),
    gender = c(rep("F", 90), rep("M", 100)),
    dataset = "lemon"
)
demog_data_chpb <- data.frame(
    age = c(rgamma(120, 95, 2), rgamma(130, 105, 2)),
    gender = c(rep("F", 120), rep("M", 130)),
    dataset = "chbp"
)

demog_data <- rbind(
  demog_data_camcan, demog_data_tuab, demog_data_lemon, demog_data_chpb)

ggplot(
  aes(x = age, color = gender, fill = gender),
      data = demog_data) +
  geom_density(alpha=0.4, size=1) +
  facet_wrap(.~dataset, ncol=2) +
  theme_minimal(base_size = 16) +
  scale_color_colorblind() +
  scale_fill_colorblind() +
  labs(x="Age [years]", y="Density")

ggsave("demographic_hypothesis.png", dpi=150, height = 6, width = 8)

out_grid <- expand.grid(
    data = c("tuab", "chbp", "lemon", "cam-can"),
    benchmarks = c("dummy", "hand-crafted", "cov-riemann", "cov-mne",
                   "deep")
)
expected_mae <- expand.grid(
    data_delta = c(0.5, -2, -1, 1),
    bench_mu = c(16, 10, 8.3, 7.5, 8.7)
)

expected_mae_cv <- cbind(out_grid, expected_mae)

CV <- 10
expected_mae_cv <- do.call(
  rbind,
  lapply(
    seq(nrow(expected_mae_cv)),
        function(ii){
          this_res <- expected_mae_cv[ii,]
          data.frame(data=this_res$data,
                     benchmark=this_res$benchmarks,
                     fold=seq(CV),
                     MAE=rgamma(
                       CV, (this_res$bench_mu + this_res$data_delta) * 2, 2))
}))

ggplot(
  aes(x = MAE, y = interaction(rev(benchmark), data), color = benchmark,
      fill = benchmark,
      shape = data) ,
  data = expected_mae_cv) +
  geom_jitter() +
  geom_boxplot(alpha=0.1) +
  theme_minimal(base_size = 16) +
  scale_color_colorblind() +
  scale_fill_colorblind() +
  scale_y_discrete(
    labels = rep(rev(c("dummy", "hand-crafted", "cov-riemann", "cov-mne",
                       "deep")), times = 5)) +
  theme(legend.position = c(.9, .5)) +
  labs(y = element_blank(), x = "age prediction MAE [10-fold CV]")

ggsave("benchmark_hypothesis.png", dpi=150, height = 6, width = 8)
