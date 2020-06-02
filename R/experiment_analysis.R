library(tidyverse)
library(knitr)
library(kableExtra)
# library(ggtext)

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
      TRUE ~ relation
    )
  )

problematic <- spp %>%
  pivot_longer(starts_with("delta"), names_to = "task", values_to = "delta") %>%
  filter(is.na(delta)) %>%
  pull(target)

results <- fs::dir_ls("data/results/") %>%
  map_df(read_csv, .id = "file") %>% 
  filter(!target %in% c("yours", "being")) %>%
  filter(!target %in% problematic) %>%
  mutate(
    model = str_extract(file, "(?<=results\\/)(.*)(?=_p)"),
    dataset = str_extract(file, "(?<=uncased_)(.*)(?=\\.csv)"),
    dataset = case_when(
      dataset == "polysemy_free" ~ "After Fix",
      TRUE ~ "Before Fix"
    ),
    dataset = factor(dataset, levels = c("Before Fix", "After Fix"))
  )

results %>%
  filter(dataset == "After Fix") %>%
  group_by(model, constraint) %>%
  summarize(
    facilitation = mean(related_probability - unrelated_probability),
    primed = mean(related_probability > unrelated_probability)
  ) %>%
  ggplot(aes(constraint, primed, group = model, fill = model, color = model)) +
  geom_col(position = "dodge") +
  scale_y_continuous(limits = c(0, 1.0)) +
  scale_color_manual(values = c("#ffb677", "#6e5773")) +
  scale_fill_manual(values = c("#ffb677", "#6e5773")) +
  labs(
    x = "Constraint",
    y = "Proportion of primed instances (n = 3084)\nFacilitation > 0"
  ) +
  theme_gray(base_size = 17) +
  theme(
    legend.position = "top"
  ) 

ggsave("paper/after_primed.png")

results %>% 
  group_by(model, dataset, constraint) %>%
  summarize(
    n = n(),
    facilitation = mean(related_probability - unrelated_probability),
    primed = mean(related_probability > unrelated_probability)
  ) %>%
  ungroup() %>%
  ggplot(aes(constraint, primed, group = model, fill = model, color = model)) +
  geom_col(position = "dodge") +
  scale_y_continuous(limits = c(0, 1.0)) +
  facet_wrap(~ dataset) +
  labs(
    x = "Constraint",
    y = "Proportion of primed\ninstances (n = 3084)"
  ) +
  theme_gray(base_size = 17) +
  theme(
    legend.position = "top"
  ) 

ggsave("paper/before_after.png")


aovs <- results %>%
  mutate(
    constraint = factor(constraint, levels = c("high", "low", "neutral")),
    facilitation = related_probability - unrelated_probability
  ) %>%
  group_by(model, dataset) %>%
  nest() %>%
  mutate(
    aov_fit = map(data, function(x) {
      aov(facilitation ~ factor(constraint), data = x)
    })
  )

summary(aovs$aov_fit[[1]])

x <- results %>%
  mutate(
    constraint = factor(constraint, levels = c("high", "low", "neutral")),
    facilitation = related_probability - unrelated_probability
  ) %>%
  filter(model == "bert-base-uncased", dataset == "After Fix")


x_aov <- aov(facilitation ~ constraint, data = x)

summary(x_aov)

levels(x$constraint)

results %>% 
  group_by(model, dataset, constraint) %>%
  summarize(
    n = n(),
    facilitation = mean(related_probability - unrelated_probability),
    facilitation_surprisal = mean(log(related_probability/unrelated_probability)),
    primed = mean(related_probability >= unrelated_probability)
  ) %>%
  ungroup() %>%
  ggplot(aes(constraint, facilitation, group = model, fill = model, color = model)) +
  geom_col(position = "dodge") +
  # scale_y_continuous(limits = c(0, 1.0)) +
  facet_wrap(~ dataset) +
  labs(
    x = "Constraint",
    y = "Change in Probability\nP(target | related) - P(target | unrelated)"
  ) +
  theme_gray(base_size = 17) +
  theme(
    legend.position = "top"
  ) 

ggsave("paper/before_after_prob.png")


results %>% 
  filter(dataset == "After Fix") %>%
  # group_by(model, constraint) %>%
  mutate(
    n = n(),
    facilitation = related_probability - unrelated_probability,
    facilitation_surprisal = mean(log(related_probability/unrelated_probability)),
    primed = mean(related_probability >= unrelated_probability)
  ) %>%
  ungroup() %>%
  ggplot(aes(constraint, facilitation, group = interaction(model, constraint), color = model)) +
  # geom_col(position = "dodge") +
  geom_jitter(position = position_jitterdodge(), alpha = 0.2) +
  geom_boxplot(position = "dodge", outlier.color = NA) +
  # scale_y_continuous(limits = c(0, 0.03)) +
  # scale_fill_manual(values = c("#5f6caf", "#ffb677")) +
  # scale_color_manual(values = c("#5f6caf", "#ffb677")) +
  # facet_wrap(~ dataset) +
  labs(
    x = "Constraint",
    y = "Facilitation, F\nP(target | related) - P(target | unrelated)"
  ) +
  theme_gray(base_size = 17) +
  theme(
    legend.position = "top"
  ) 

ggsave("paper/after_prob.png")


spp %>%
  filter(!target %in% problematic) %>%
  pivot_longer(starts_with("delta"), names_to = "task", values_to = "delta") %>%
  group_by(task) %>%
  mutate(
    delta_standardized = scale(delta)
  )




spp %>% 
  filter(!target %in% problematic) %>%
  pivot_longer(starts_with("delta"), names_to = "task", values_to = "delta") %>%
  ggplot(aes(delta)) + 
  geom_density() + 
  facet_wrap(~task)


naive_lm <- results %>%
  mutate(facilitation = related_probability - unrelated_probability) %>%
  select(model, dataset, constraint, target, related, unrelated, facilitation) %>%
  inner_join(
    spp %>%
      mutate(
        related = str_to_lower(related),
        unrelated = str_to_lower(unrelated)
      ) %>%
      filter(!target %in% problematic) %>%
      pivot_longer(starts_with("delta"), names_to = "task", values_to = "delta")
  ) %>%
  group_by(model, dataset, task) %>%
  nest() %>%
  mutate(
    fit = map(data, ~lm(delta ~ scale(facilitation), .x)),
    glanced = map(fit, broom::glance)
  ) %>%
  unnest(glanced)


results %>%
  mutate(facilitation = -log2(related_probability) + log2(unrelated_probability)) %>%
  select(model, dataset, constraint, target, related, unrelated, facilitation) %>%
  inner_join(
    spp %>%
      mutate(
        related = str_to_lower(related),
        unrelated = str_to_lower(unrelated)
      ) %>%
      filter(!target %in% problematic) %>%
      pivot_longer(starts_with("delta"), names_to = "task", values_to = "delta")
  ) %>%
  filter(dataset == "After Fix") %>%
  group_by(model, constraint, task) %>%
  nest() %>%
  mutate(
    correlation = map_dbl(data, function(x) {
      cor(x$delta, x$facilitation)
    })
  ) %>% select(-data) %>% View()


top_relations <- spp %>%
  mutate(
    related = str_to_lower(related),
    unrelated = str_to_lower(unrelated)
  ) %>%
  filter(!target %in% problematic) %>%
  count(relation) %>%
  filter(!is.na(relation), relation != "unclassified") %>%
  top_n(10, n)

rel_levels <- top_relations %>% 
  arrange(-n) %>% 
  mutate(relation = paste0(relation, "(", n, ")")) %>% 
  pull(relation)

results %>%
  inner_join(
    spp %>%
      filter(!target %in% problematic) %>%
      select(target, related, unrelated, relation)
  ) %>%
  group_by(model, dataset, constraint, relation) %>%
  summarize(
    primed = mean(related_probability > unrelated_probability)
  ) %>%
  filter(relation %in% top_relations$relation) %>%
  inner_join(top_relations) %>%
  mutate(
    relation = paste0(relation, "(", n, ")"),
    relation = factor(relation, levels = rel_levels)
  ) %>%
  filter(dataset == "After Fix") %>%
  ggplot(aes(constraint, primed, group = model, color = model, fill = model)) +
  geom_col(position = "dodge") +
  facet_wrap(~ relation, ncol = 3) +
  scale_fill_manual(values = c("#5f6caf", "#ffb677")) +
  scale_color_manual(values = c("#5f6caf", "#ffb677")) +
  theme_gray(base_size = 15) +
  theme(
    legend.position = "top",
    panel.grid.major=element_blank(),
    panel.grid.minor=element_blank()
  ) +
  labs(
    x = "Constraint",
    y = "Proportion of primed instances"
  )

ggsave("paper/relationpriming.png")

results %>%
  inner_join(
    spp %>%
      filter(!target %in% problematic) %>%
      select(target, related, unrelated, relation)
  ) %>%
  group_by(model, dataset, constraint, relation) %>%
  summarize(
    primed = mean(related_probability - unrelated_probability)
  ) %>%
  filter(relation %in% top_relations$relation) %>%
  inner_join(top_relations) %>%
  mutate(
    relation = paste0(relation, "(", n, ")"),
    relation = factor(relation, levels = rel_levels)
  ) %>%
  filter(dataset == "After Fix") %>%
  ggplot(aes(constraint, primed, group = model, color = model, fill = model)) +
  geom_col(position = "dodge") +
  facet_wrap(~ relation, ncol = 3) +
  scale_fill_manual(values = c("#5f6caf", "#ffb677")) +
  scale_color_manual(values = c("#5f6caf", "#ffb677")) +
  theme_gray(base_size = 15) +
  theme(
    legend.position = "top",
    panel.grid.major=element_blank(),
    panel.grid.minor=element_blank()
  ) +
  labs(
    x = "Constraint",
    y = "Change in Probability\nP(target | related) - P(target | unrelated)"
  )

ggsave("paper/relation_probability.png")

results %>%
  filter(dataset == "After Fix") %>%
  group_by(model, constraint) %>%
  nest() %>%
  mutate(
    t_test = map(data, function(x) {
      wilcox.test(x$related_probability, x$unrelated_probability) %>% broom::glance()
    })
  ) %>% unnest(t_test)

x <- results %>% filter(dataset == "After Fix", model == "bert-base-uncased", constraint == "neutral")

wilcox.test(x$related_probability, x$unrelated_probability, alternative = "greater", conf.int = TRUE) %>% broom::glance()    

results %>%
  filter(dataset == "After Fix") %>%
  group_by(model, constraint) %>%
  summarize(
    facilitation = mean(log2(unrelated_probability/related_probability)),
    # sd = sd(related_probability - unrelated_probability),
    sd = sd(log2(unrelated_probability/related_probability)),
    # primed = mean(related_probability > unrelated_probability) %>% scales::percent()
    primed = mean(log2(unrelated_probability/related_probability) < 0) %>% scales::percent()
  ) %>%
  kable("latex", digits = 2)

results %>%
  filter(dataset == "After Fix") %>%
  # select(category, model, constraint, target, related, unrelated, related_probability, unrelated_probability) %>%
  mutate(
    facilitation = related_probability - unrelated_probability
  ) %>%
  group_by(model, constraint) %>%
  filter(facilitation == max(facilitation)) %>% View()

results %>%
  filter(dataset == "After Fix") %>%
  # select(category, model, constraint, target, related, unrelated, related_probability, unrelated_probability) %>%
  mutate(
    facilitation = related_probability - unrelated_probability
  ) %>%
  group_by(model, constraint) %>%
  filter(facilitation == min(facilitation)) %>% View()


results %>%
  filter(dataset == "After Fix") %>%
  select(category, model, constraint, target, related, unrelated, related_probability, unrelated_probability) %>%
  mutate(
    facilitation = related_probability - unrelated_probability
  )

results %>%
  inner_join(
    spp %>%
      filter(!target %in% problematic) %>%
      select(target, related, unrelated, relation)
  ) %>%
  # group_by(model, dataset, constraint, relation) %>%
  mutate(
    primed = related_probability - unrelated_probability
  ) %>%
  filter(relation %in% top_relations$relation) %>%
  inner_join(top_relations) %>%
  mutate(
    relation = paste0(relation, "(", n, ")"),
    relation = factor(relation, levels = rel_levels)
  ) %>%
  filter(dataset == "After Fix") %>%
  ggplot(aes(constraint, primed, group = interaction(model, constraint), color = model, fill = model)) +
  geom_jitter(position = position_jitterdodge(), alpha = 0.3) +
  geom_boxplot(position = "dodge", outlier.color = NA) +
  facet_wrap(~ relation, ncol = 3) +
  scale_fill_manual(values = c("#5f6caf", "#ffb677")) +
  scale_color_manual(values = c("#5f6caf", "#ffb677")) +
  theme_gray(base_size = 15) +
  theme(
    legend.position = "top",
    panel.grid.major=element_blank(),
    panel.grid.minor=element_blank()
  ) +
  labs(
    x = "Constraint",
    y = "Facilitation, F\nP(target | related) - P(target | unrelated)"
  )

ggsave("paper/after_relations_prob.png")


results %>%
  # inner_join(spp %>% select(target, related, unrelated, association)) %>%
  filter(dataset == "After Fix") %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  # group_by(model, constraint, association) %>%
  group_by(model, constraint) %>%
  summarize(
    n = n(),
    avg_facilitation = mean(facilitation),
    high = avg_facilitation + sd(facilitation),
    low = avg_facilitation - sd(facilitation)
  ) %>%
  ggplot(aes(constraint, avg_facilitation, group = model, fill = model, color = model)) +
  geom_col(position = "dodge") + 
  geom_errorbar(aes(ymin = low, ymax = high), color = "black", position = "dodge") +
  labs(
    x = "Constraint",
    y = "Facilitation, F\nSurp(T | R) - Surp(T | U)"
  ) +
  theme_gray(base_size = 17) +
  theme(
    legend.position = "top"
  ) 
  # facet_wrap(~association)



results %>%
  inner_join(
    spp %>%
      filter(!target %in% problematic) %>%
      select(target, related, unrelated, relation)
  ) %>%
  # group_by(model, dataset, constraint, relation) %>%
  mutate(
    facilitation = -log2(related_probability/unrelated_probability)
  ) %>%
  filter(relation %in% top_relations$relation) %>%
  inner_join(top_relations) %>%
  mutate(
    relation = paste0(relation, "(", n, ")"),
    relation = factor(relation, levels = rel_levels)
  ) %>%
  filter(dataset == "After Fix") %>%
  group_by(model, constraint, relation) %>%
  summarize(
    n = n(),
    f = mean(facilitation),
    high = f + 1.96 * plotrix::std.error(facilitation),
    low = f - 1.96 * plotrix::std.error(facilitation)
  ) %>%
  ggplot(aes(constraint, f, group = model, fill = model, color = model)) +
  geom_col(position = "dodge") + 
  geom_errorbar(aes(ymin = low, ymax = high), color = "black", width = .5, position = position_dodge(width = 0.9)) +
  facet_wrap(~relation, ncol = 5) +
  labs(
    x = "Context Type",
    y = "Facilitation, F (measured in bits)\nSurp(T | R, C) - Surp(T | U, C)"
  ) +
  theme_gray(base_size = 17) +
  theme(
    legend.position = "top",
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

ggsave("paper/relation_surprisal.png", height = 6, width = 15)
ggsave("paper/relationsurprisals.pdf", height = 6, width = 14)  


results %>%
  filter(dataset == "After Fix") %>%
  mutate(
    facilitation = log2(unrelated_probability/related_probability)
  ) %>%
  group_by(model, constraint) %>%
  summarize(
    facilitation = mean(facilitation)
  )
