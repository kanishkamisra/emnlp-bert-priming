library(tidyverse)

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

bb <- read_csv("data/entropies_bert-base-uncased.csv") %>% pull(entropy)
bl <- read_csv("data/entropies_bert-base-uncased.csv") %>% pull(entropy)

contexts <- read_csv("data/unique_contexts.csv") %>%
  select(context = isolated_context, constraint) %>%
  mutate(
    entropy = (bb + bl)/2
  )

cor.test(contexts$constraint, contexts$entropy)

raw_results <- bucket_results %>%
  inner_join(
    contexts %>%
      select(isolated_context = context, constraint, entropy)
  ) %>% 
  mutate(facilitation = -log2(unrelated_probability/related_probability)) %>%
  distinct(target, related, unrelated, category, constraint, model, context_type, entropy, facilitation)

raw_results %>%
  filter(category != 0) %>%
  ggplot(aes(entropy, facilitation, color = context_type)) +
  geom_point(alpha = 0.1) +
  geom_smooth(method = "lm") +
  facet_grid(context_type ~ model)


raw_results %>%
  group_by(model, category) %>%
  summarize(
    entropy = mean(entropy)
  ) %>%
  ungroup() %>%
  filter(model == "bert-base-uncased") %>%
  ggplot(aes(category, entropy)) +
  geom_col(fill = "#158467") +
  scale_x_continuous(breaks = 0:10) +
  theme_bw(base_size = 17) +
  theme(
    legend.position = "top",
    panel.grid.minor.x = element_blank()
  ) +
  labs(
    y = "Average Entropy (in bits)",
    x = "Constraint Bin"
  )


raw_results %>%
  group_by(model, context_type) %>%
  nest() %>%
  mutate(
    fit = map(data, function(x){
      broom::glance(lm(x$facilitation ~ x$entropy))
    })
  ) %>%
  unnest(fit)


contexts %>% 
  ggplot(aes(constraint, entropy)) + 
  geom_point(alpha = 0.08, size = 2) +
  geom_smooth(method = "lm", se = TRUE) +
  theme_bw(base_size = 18) +
  labs(
    x = "Constraint Score",
    y = "Entropy (in bits)"
  )

ggsave("thesis/entropy_vs_constraint.pdf")
