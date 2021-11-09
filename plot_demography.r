library(ggplot2)
library(scales)
library(ggthemes)   
source("./utils.r")

demog_data <- read.csv("./outputs/demog_summary.csv")
demog_data <- subset(demog_data, sex != "")
demog_data$sex <- factor(demog_data$sex, levels = c("M", "F"))
demog_data$dataset <- factor(
  demog_data$dataset, levels = c("camcan", "lemon", "chbp", "tuab"),
  labels = c("Cam-CAN\n(MEG)", "LEMON\n(EEG)", "CHBP\n(EEG)", "TUAB\n(EEG)"))


demog_summary1 <- do.call(rbind, lapply(
  split(demog_data, list(demog_data$sex, demog_data$dataset)),
  function(dat){
    with(dat,
      data.frame(dataset = dat$dataset[[1]],
                 sex = sex[[1]],
                 sub_count = length(sex),
                 sub_M = mean(age),
                 sub_min = min(age),
                 sub_max = max(age))
    )
}))

demog_summary2 <- do.call(rbind, lapply(
  split(demog_data, demog_data$dataset),
  function(dat) {
    with(dat,
      data.frame(dataset = dat$dataset[[1]],
                 count = nrow(dat),
                 min = min(age),
                 max = max(age),
                 SD = sd(age),
                 M = mean(age))
    )
}))

demog_out <- merge(demog_summary2, demog_summary1, by = "dataset")

is_num <- sapply(demog_out, is.numeric)
demog_out[, is_num] <- round(demog_out[, is_num], 1)

write.csv(demog_out, "./outputs/demog_summary_table.csv")

demog_count <- demog_out[seq(1, 8, 2), c("dataset", "count")]
demog_count$label <- paste('n =', demog_count$count)

fig <- ggplot(
  aes(x = age, color = sex, fill = sex),
      data = demog_data) +
  geom_density(alpha = 0.4, size = 1, trim = T) +
  geom_rug() +
  facet_wrap(.~dataset, ncol = 4) +
  theme_minimal(base_size = 22) +
  scale_color_solarized() +
  scale_fill_solarized() +
  labs(x = "Age [years]", y = "Density") +
  geom_text(x = 70, y = 0.026,
            size = 6,
            inherit.aes = F,
            mapping = aes(label =  label), data = demog_count)
print(fig)
my_ggsave("./figures/fig_demographics", fig, dpi = 300, width = 10, height = 4.5)