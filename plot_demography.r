library(ggplot2)
library(scales)
library(ggthemes)   
source('./utils.r')

demog_data <- read.csv('./outputs/demog_summary.csv')
demog_data <- subset(demog_data, sex != "")
demog_data$sex <- factor(demog_data$sex, levels = c('M', 'F')) 
demog_data$dataset <- factor(demog_data$dataset,
                             levels = c('camcan', 'lemon', 'tuab', 'chbp'))

fig <- ggplot(
  aes(x = age, color = sex, fill = sex),
      data = demog_data) +
  geom_density(alpha=0.4, size=1, trim = T) +
  geom_rug() +
  facet_wrap(.~dataset, ncol=4) +
  theme_minimal(base_size = 22) +
  scale_color_solarized() +
  scale_fill_solarized() +
  labs(x="Age [years]", y="Density")

my_ggsave('./figures/fig_demographics', fig, dpi = 300, width = 10, height = 4)

demog_summary <- do.call(rbind, lapply(
  split(demog_data, list(demog_data$sex, demog_data$dataset)),
  function(dat){
    with(dat, 
      data.frame(dataset = dat$dataset[[1]],
                 sub_count = length(sex),
                 sex = sex[[1]],
                 age = mean(age))
    )
}))

demog_out <- merge(
  demog_summary,
  aggregate(sub_count ~ dataset, data = demog_summary, FUN = sum),
  by = 'dataset'
)

names(demog_out) <- c('dataset', 'sub_count', 'sex', 'age', 'count')

write.csv(demog_out, './outputs/demog_summary_table.csv')
