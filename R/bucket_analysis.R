library(tidyverse)
library(patchwork)
library(lme4)

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

## Subset of instances, for maybe 2 targets?

bucket_results %>%
  filter(dataset == "word", model == "bert-base-uncased") %>%
  mutate(
    facilitation = -log2(unrelated_probability/related_probability)
  ) %>%
  select(category, target, related, unrelated, isolated_context, facilitation) %>% 
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  group_by(target, related, unrelated) %>%
  nest() %>%
  mutate(
    corr = map_dbl(data, function(x) {
      cor(x %>% arrange(category) %>% pull(facilitation), 10:0)
    })
  ) %>%
  unnest() %>%
  View()

cherries_t <- c("spring", "branch", "apart")
cherries_r <- c("autumn", "twig", "together")

bucket_results %>%
  filter(target %in% cherries_t, related %in% cherries_r) %>%
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  mutate(
    facilitation = -log2(unrelated_probability/related_probability)
  ) %>%
  select(model, scenario = dataset, bucket = category, constraint, target, related, unrelated, relation, unprimed_context = isolated_context, related_context, unrelated_context, facilitation) %>%
  arrange(bucket, desc(scenario), model) %>%
  write_csv("data/example_data.csv")

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

neutral_f <- facilitations %>%
  filter(bucket == 0)

p_facilitations <- facilitations %>%
  filter(bucket != 0) %>%
  ggplot(aes(bucket/10, facilitation, color = context_type, fill = context_type)) +
  geom_point(size = 2) +
  geom_point(data = neutral_f, aes(bucket/10, facilitation), shape = 17, size = 5, show.legend = FALSE) +
  geom_line(size = 0.9) +
  # geom_errorbar(aes(ymin = low_f, ymax = high_f), width = 0.05) +
  geom_ribbon(aes(ymin = low_f, ymax = high_f), alpha = 0.2, color = NA) +
  scale_x_continuous(limits = c(0, 1)) +
  # scale_y_continuous(limits = c(0, 2)) +
  scale_color_manual(values = c("#005082", "#f2a365")) + 
  scale_fill_manual(values = c("#005082", "#f2a365")) +
  facet_wrap(~model) +
  labs(
    x = "",
    y = "Facilitation (in bits)",
    color = "Scenario",
    fill = "Scenario"
  ) +
  ggtitle("(A)") +
  theme_gray(base_size = 20) +
  theme(
    legend.position = "top",
    axis.title.x = element_blank(),
    plot.title = element_text(vjust = -7, hjust = -0.1),
    plot.margin = margin(t = -10),
    panel.grid.minor.y = element_blank()
  )

p_facilitations

ggsave("paper/bins_vs_priming.png")
ggsave("paper/bins_vs_priming.pdf")


bucket_results %>%
  rename(bucket = "category") %>%
  filter(bucket != 0) %>%
  group_by(model, context_type, dataset) %>%
  mutate(
    isolated_surprisal = -log2(isolated_probability),
    f = log2(unrelated_probability/related_probability),
    rf = log2(isolated_probability/related_probability)
  ) %>%
  nest() %>%
  mutate(
    fit = map(data, ~lm(f ~ constraint, data = .x)), 
    glanced = map(fit, broom::glance)
  ) %>%
  unnest(glanced) %>%
  arrange(-r.squared)



bucket_results %>%
  rename(bucket = "category") %>%
  group_by(model, context_type, dataset) %>%
  mutate(
    isolated_surprisal = -log2(isolated_probability),
    f = log2(unrelated_probability/related_probability),
    rf = log2(isolated_probability/related_probability)
  ) %>% 
  filter(model == "bert-base-uncased", dataset == "word", bucket != 0) %>%
  ggplot(aes(isolated_surprisal, -log2(unrelated_probability), group = bucket, color = bucket)) +
  geom_point() +
  geom_smooth() +
  facet_wrap(~bucket)



p_facilitated <- facilitations %>%
  filter(bucket != 0) %>%
  ggplot(aes(bucket/10, facilitated, color = context_type)) +
  geom_point(size = 3) +
  geom_point(data = neutral_f, aes(bucket/10, facilitated), shape = 17, size = 5, show.legend = FALSE) +
  geom_line(size = 1) +
  # geom_errorbar(aes(ymin = low_f, ymax = high_f)) +
  # geom_ribbon(aes(fill = model, ymin = low_f, ymax = high_f), alpha = 0.2) +
  # scale_x_continuous(breaks = seq(0, 1, by = .1), limits = c(0, 1.0)) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_colour_manual(values = c("#005082", "#f2a365")) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.0)) +
  facet_wrap(~model) +
  labs(
    x = "Binned Constraint score (Probability of most expected word)",
    y = "Proportion of primed instances\n(F > 0)",
    color = "Scenario"
  ) +
  ggtitle("(B)") +
  theme_gray(base_size = 20) +
  theme(
    legend.position = "none",
    plot.title = element_text(vjust = -3, hjust = -0.1),
    plot.margin = margin(t = -1),
    axis.title.y = element_text(size = 17),
    axis.title.x = element_text(size = 17)
    # panel.grid.minor.x =element_blank()
  )

p_facilitated

ggsave("paper/bins_vs_primed_instances.pdf")
ggsave("paper/bins_vs_primed_instances.png")


p_facilitations / p_facilitated

ggsave("paper/overallfacilitations2.pdf")

relations <- bucket_results %>%
  inner_join(
    spp %>% 
      select(target, related, unrelated, relation)
  ) %>% distinct(target, related, unrelated, relation) %>% count(relation, sort = TRUE) %>% filter(!relation %in% c("unclassified", NA)) %>%
  top_n(10, n)

rel_levels <- relations %>% 
  arrange(-n) %>% 
  mutate(relation = paste0(relation, " (", n, ")")) %>% 
  pull(relation)

relation_facilitation <- bucket_results %>%
  rename(bucket = "category") %>%
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  # filter(bucket != 0) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  select(model, bucket, relation, context_type, target, related, unrelated, facilitation) %>%
  group_by(bucket, context_type, relation) %>%
  summarize(
    f = mean(facilitation),
    high = f + 1.96 * plotrix::std.error(facilitation),
    low = f - 1.96 * plotrix::std.error(facilitation)
  ) %>%
  inner_join(relations) %>%
  mutate(
    relation = paste0(relation, " (", n, ")"), 
    relation = factor(relation, levels = rel_levels)
  ) %>%
  ungroup()

neutral_relation_f <- relation_facilitation %>%
  filter(bucket == 0)

relation_facilitation %>%
  filter(bucket != 0) %>%
  ggplot(aes(bucket/10, f, color = context_type, fill = context_type)) +
  geom_point(size = 2) +
  geom_hline(data = neutral_relation_f, aes(yintercept = f, color = context_type), linetype = 2, size = 1) +
  # geom_point(data = neutral_relation_f, size = 3, shape = 17, show.legend = FALSE) +
  geom_line()+
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.2, color = NA) +
  facet_wrap(~relation, nrow = 2) +
  scale_x_continuous(limits = c(0, 1)) +
  # scale_y_continuous(limits = c(0, 3)) +
  scale_color_manual(values = c("#005082", "#f2a365")) + 
  scale_fill_manual(values = c("#005082", "#f2a365")) +
  theme_gray(base_size = 14) +
  theme(
    legend.position = "top",
    # strip.text = element_text(face = "bold"),
    # axis.title = element_text(face = "bold"),
    # legend.title = element_text(face = "bold"),
    # plot.title = element_text(face = "bold"),
    panel.grid.minor.y=element_blank()
  ) +
  labs(
    x = "Binned Constraint score (Probability of most expected word)",
    y = "Facilitation (in bits)",
    color = "Prime Context",
    fill = "Prime Context"
  )

ggsave("paper/relationpriming2.pdf", width = 13.4, height = 5.9)
ggsave("paper/relationpriming.png", height = 5.9, width = 13.4)


bucket_results %>%
  rename(bucket = "category") %>%
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  mutate(
    facilitation = -log2(related_probability/unrelated_probability)
  ) %>%
  select(model, constraint, relation, context_type, target, related, unrelated, facilitation) %>%
  inner_join(relations) %>%
  mutate(
    relation = paste0(relation, " (", n, ")"), 
    relation = factor(relation, levels = rel_levels)
  ) %>%
  ungroup() %>%
  filter(context_type == "word", constraint != 0, model == "bert-base-uncased") %>%
  ggplot(aes(constraint, facilitation, group = model)) +
  geom_point() +
  geom_smooth(method = "lm") +
  # geom_line()+
  facet_wrap(~relation, nrow = 2) +
  theme_gray(base_size = 14) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    # panel.grid.minor=element_blank()
  ) +
  labs(
    x = "Constraint score (Probability of most expected word)\n Neutral Contexts = 0.0",
    y = "Facilitation, F (measured in bits)\nSurp(T | R, C) - Surp(T | U, C)",
    color = "Prime Context"
  )

word_base <- bucket_results %>%
  rename(bucket = "category") %>%
  # inner_join(
  #   spp %>%
  #     select(target, related, unrelated, relation) 
  # ) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  # select(model, constraint, relation, context_type, target, related, unrelated, facilitation) %>%
  select(model, constraint, bucket, context_type, target, related, unrelated, facilitation) %>%
  # inner_join(relations) %>%
  # mutate(
  #   relation = paste0(relation, " (", n, ")"), 
  #   relation = factor(relation, levels = rel_levels)
  # ) %>%
  ungroup() %>%
  filter(context_type == "word", constraint != 0, model == "bert-base-uncased")

lmer_fit <- lmerTest::lmer(facilitation ~ scale(constraint, scale = FALSE) + (1 | target), data = word_base)

summary(lmer_fit)

word_base %>% 
  filter(target == "ability") %>%
  ggplot(aes(bucket, facilitation)) +
  geom_point() +
  geom_smooth(method = "lm")


bucket_results %>%
  rename(bucket = "category") %>%
  mutate(facilitation = log2(unrelated_probability/related_probability)) %>%
  group_by(bucket, model, dataset) %>%
  summarize(
    facilitated = mean(facilitation < 0), # priming observed
    worse = mean(isolated_probability > related_probability)
  ) %>%
  gather(facilitated, worse, key = "metric", value = "proportion") %>%
  mutate(
    metric = case_when(
      metric == "facilitated" ~ "P(T | R) > P(T | U)",
      metric == "worse" ~ "P(T | R) < P(T)"
    )
  ) %>%
  ggplot(aes(bucket/10, proportion, shape = dataset, color = metric)) +
  geom_point(size = 3) +
  geom_line(size = 1) +
  # geom_errorbar(aes(ymin = low_f, ymax = high_f)) +
  # geom_ribbon(aes(fill = model, ymin = low_f, ymax = high_f), alpha = 0.2) +
  # scale_x_continuous(breaks = seq(0, 1, by = .1), limits = c(0, 1.0)) +
  scale_color_manual(values = c("#4b97a8", "#db701e")) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.0)) +
  facet_wrap(~model) +
  labs(
    x = "Constraint score (Probability of most expected word)\n Neutral Contexts = 0.0",
    y = "Proportion of Instances (n = 2118)",
    color = "Metric",
    shape = "Prime Context"
  ) +
  theme_gray(base_size = 17) +
  theme(
    legend.position = "top",
    # panel.grid.minor.x =element_blank()
  )

ggsave("paper/priming_and_distraction.png")  
ggsave("paper/priming_and_distraction.pdf")  


bucket_results %>%
  rename(bucket = "category") %>%
  # inner_join(
  #   spp %>%
  #     select(target, related, unrelated, relation) 
  # ) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  # select(model, constraint, relation, context_type, target, related, unrelated, facilitation) %>%
  select(model, constraint, bucket, context_type, target, related, unrelated, facilitation) %>%
  filter(bucket != 0) %>%
  # inner_join(relations) %>%
  # mutate(
  #   relation = paste0(relation, " (", n, ")"), 
  #   relation = factor(relation, levels = rel_levels)
  # ) %>%
  group_by(model, context_type) %>%
  nest() %>%
  mutate(
    lmer_fit = map(data, function(x) {
      fit <- lmerTest::lmer(facilitation ~ scale(constraint, scale = FALSE) + (1 | target), data = x)
      summary_fit <- summary(fit)
      summary_fit$coefficients
    })
  ) %>% pull(lmer_fit)

bucket_results %>%
  rename(bucket = "category") %>%
  # inner_join(
  #   spp %>%
  #     select(target, related, unrelated, relation) 
  # ) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  # select(model, constraint, relation, context_type, target, related, unrelated, facilitation) %>%
  select(model, constraint, bucket, context_type, target, related, unrelated, facilitation) %>% pivot_wider(names_from = context_type, values_from = facilitation, names_prefix = "facilitation_") %>%
  group_by(model, bucket) %>%
  nest() %>%
  mutate(
    t_test = map(data, function(x){
      broom::tidy(t.test(x$facilitation_sentence, x$facilitation_word, alternative = "greater"))
    })
  ) %>%
  unnest(t_test) %>% select(-data)


bucket_results %>%
  rename(bucket = "category") %>%
  # inner_join(
  #   spp %>%
  #     select(target, related, unrelated, relation) 
  # ) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  # select(model, constraint, relation, context_type, target, related, unrelated, facilitation) %>%
  select(model, constraint, bucket, context_type, target, related, unrelated, facilitation) %>% pivot_wider(names_from = model, values_from = facilitation, names_repair = "universal") %>%
  group_by(context_type, bucket) %>%
  nest() %>%
  mutate(
    t_test = map(data, function(x){
      broom::tidy(t.test(x$bert.large.uncased, x$bert.base.uncased, alternative = "less"))
    })
  ) %>%
  unnest(t_test) %>% select(-data) %>% View()

bucket_results %>%
  rename(bucket = "category") %>%
  mutate(facilitation = log2(unrelated_probability/related_probability)) %>%
  group_by(bucket, model, dataset) %>%
  summarize(
    pre = mean(facilitation < 0),
    post = mean(facilitation < 0 & related_probability > isolated_probability)
  ) %>%
  gather(pre, post, key = "metric", value = "proportion") %>%
  filter(bucket != 0) %>%
  mutate(
    metric = case_when(
      metric == "pre" ~ "F > 0",
      metric == "post" ~ "F > 0 and\nP(T | R, C) > P(T | C)"
    )
  ) %>%
  ggplot(aes(bucket/10, proportion, color = dataset, linetype = metric)) +
  geom_point(size = 3.5) +
  geom_line(size = 1) +
  geom_hline(yintercept = 0.5, linetype = 2) +
  annotate("text", x = 0.25, y = 0.25, label = "Majority of Related \n Primes are Distractors", color = "black", size = 5) +
  # geom_errorbar(aes(ymin = low_f, ymax = high_f)) +
  # geom_ribbon(aes(fill = model, ymin = low_f, ymax = high_f), alpha = 0.2) +
  # scale_x_continuous(breaks = seq(0, 1, by = .1), limits = c(0, 1.0)) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_colour_manual(values = c("#005082", "#f2a365")) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.0)) +
  guides(
    color = guide_legend(
      title.position = "left",
      direction = "vertical",
      hjust = -1
    ),
    linetype = guide_legend(
      title.position = "left",
      direction = "vertical"
    )
  ) +
  facet_wrap(~model, nrow = 2) +
  labs(
    x = "Binned Constraint score\n(Probability of most expected word)",
    y = "Proportion of primed instances",
    color = "Scenario",
    linetype = "Criterion"
  ) +
  theme_gray(base_size = 15) +
  theme(
    legend.position = "top",
    # legend.position = c(0, 1),
    # legend.justification = c(-1, 1),
    axis.text = element_text(size = 17),
    strip.text = element_text(size = 15),
    legend.text = element_text(size = 13),
    legend.title = element_text(size = 14)
    # panel.grid.minor.x =element_blank()
  )

ggsave("paper/postdistractor2.pdf")


by_relation <- bucket_results %>%
  rename(bucket = "category") %>%
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  filter(bucket != 0, relation %in% relations$relation) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  select(model, constraint, relation, context_type, target, related, unrelated, facilitation) %>%
  group_by(model, context_type, relation) %>%
  nest() %>%
  mutate(
    lmer_fit = map_dbl(data, function(x) {
      fit <- lmerTest::lmer(facilitation ~ scale(constraint, scale = FALSE) + (1 | target), data = x)
      summary_fit <- summary(fit)
      summary_fit$coefficients[, 1][2]
    })
  )

special_targets <- bucket_results %>%
  # mutate(distraction = related_probability - isolated_probability)
  filter(category > 7, dataset == "word") %>%
  filter(unrelated_rank != 1) %>%
  distinct(target)

bucket_results %>%
  mutate(distraction = related_probability - isolated_probability) %>%
  filter(constraint > 0.7, model == "bert-large-uncased", dataset == "word") %>%
  select(constraint, target, related, unrelated, isolated_context, distraction) %>%
  View()

