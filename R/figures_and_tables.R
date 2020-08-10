library(tidyverse)
library(scales)

spp <- read_csv("data/spp.csv") %>%
  select(-context_word) %>%
  rename(related = "prime") %>%
  distinct() %>%
  mutate(
    target = str_to_lower(target),
    relation = str_to_lower(relation),
    relation = case_when(
      str_detect(relation, "uncl") ~ "unclassified",
      str_detect(relation, "anton") ~ "antonym",
      relation == "fpa" ~ "forward phrasal associate",
      relation == "bpa" ~ "backward phrasal associate",
      TRUE ~ relation
    )
  )

same <- spp %>%
  filter(related == unrelated) %>%
  distinct(target) %>%
  pull(target)

bucket_results <- fs::dir_ls("data/results/", glob = "*_10.csv") %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_extract(file, "(?<=results\\/)(.*)(?=\\.csv)"),
    context_type = case_when(
      str_detect(model, "sentence") ~ "sentence",
      TRUE ~ "word"
    ),
    model = str_replace(model, "(_sentence_10|_word_10)", ""),
    dataset = str_extract(file, "(?<=uncased_)(.*)(?=\\_10)")
  ) %>%
  filter(!target %in% same)

bucket_results %>%
  distinct(category, constraint, target, related, unrelated) %>%
  group_by(category) %>%
  summarize(constraint = mean(constraint)) %>%
  filter(category != 0) %>%
  ggplot(aes(category, constraint)) +
  geom_col() +
  scale_y_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(8)) +
  scale_x_continuous(breaks = 1:10) +
  theme_bw(base_size = 17) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "top"
  ) +
  labs(
    y = "Constraint Score"
  )

facilitations <- bucket_results %>%
  rename(bucket = "category") %>%
  group_by(model, bucket, context_type, dataset) %>%
  mutate(
    f = -log2(unrelated_probability/related_probability),
    rf = log2(isolated_probability/related_probability)
  ) %>%
  summarize(
    facilitation = mean(f),
    se_f = 1.96 * plotrix::std.error(f),
    # se_f = sd(f),
    high_f = facilitation + se_f,
    low_f = facilitation - se_f,
    facilitated = mean(f > 0)
  ) %>%
  ungroup()
