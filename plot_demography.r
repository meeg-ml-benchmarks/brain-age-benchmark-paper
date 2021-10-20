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
