library(tidyverse)
library(scales)
library(lme4)
library(broom)

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

spp

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

raw_results <- bucket_results %>%
  inner_join(
    contexts %>%
      select(isolated_context = context, constraint, entropy)
  ) %>% 
  mutate(facilitation = -log2(unrelated_probability/related_probability)) %>%
  distinct(target, related, unrelated, category, constraint, model, context_type, entropy, facilitation)

bucket_results %>%
  distinct(category, constraint, target, related, unrelated) %>%
  group_by(category) %>%
  summarize(constraint = mean(constraint)) %>%
  filter(category != 0) %>%
  ggplot(aes(category, constraint)) +
  geom_col(fill = "#900c3f", color = "#900c3f") +
  scale_y_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(8)) +
  scale_x_continuous(breaks = 1:10) +
  theme_bw(base_size = 18) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "top"
  ) +
  labs(
    x = "Constraint Bin",
    y = "Constraint Score"
  )

ggsave("thesis/avgconstraint2.pdf", height = 5.96, width = 6.55)


# Facilitation > 0 t-test
raw_facilitations <- bucket_results %>%
  rename(bucket = "category") %>%
  group_by(model, bucket, context_type, dataset) %>%
  mutate(
    f = -log2(unrelated_probability/related_probability),
    rf = log2(isolated_probability/related_probability)
  )

raw_facilitations %>%
  nest() %>%
  mutate(
    ttest = map(data, function(x) {
      broom::tidy(t.test(x$f, alternative = "greater"))
    })
  ) %>%
  unnest(ttest) %>%
  select(-data) %>%
  View()

# summarized
facilitations <- raw_facilitations %>%
  summarize(
    facilitation = mean(f),
    se_f = 1.96 * plotrix::std.error(f),
    # se_f = sd(f),
    high_f = facilitation + se_f,
    low_f = facilitation - se_f,
    facilitated = mean(f > 0)
  ) %>%
  ungroup()

facilitations

contexts %>% 
  ggplot(aes(constraint, entropy)) + 
  geom_point(alpha = 0.08, size = 2) +
  geom_smooth(method = "lm", se = TRUE) +
  theme_bw(base_size = 18) +
  labs(
    x = "Constraint Score",
    y = "Entropy (in bits)"
  )

ggsave("thesis/entropyconstraint.pdf", height = 5.96, width = 6.55)



# Facilitation vs Constraint (for non Neutral constraints)

facilitations %>%
  filter(bucket != 0) %>%
  ggplot(aes(bucket/10, facilitation, color = context_type, fill = context_type)) +
  geom_point(size = 3) +
  # geom_point(data = neutral_f, aes(bucket/10, facilitation), shape = 17, size = 5, show.legend = FALSE) +
  geom_line(size = 0.9) +
  # geom_errorbar(aes(ymin = low_f, ymax = high_f), width = 0.05) +
  geom_ribbon(aes(ymin = low_f, ymax = high_f), alpha = 0.2, color = NA) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 2)) +
  scale_color_manual(values = c("#005082", "#f2a365")) + 
  scale_fill_manual(values = c("#005082", "#f2a365")) +
  facet_wrap(~model) +
  labs(
    x = "Binned Constraint Score (Probability of most expected word)",
    y = "Facilitation (in bits)",
    color = "Scenario",
    fill = "Scenario"
  ) +
  # ggtitle("(A)") +
  theme_bw(base_size = 20) +
  theme(
    legend.position = "top",
    # axis.title.x = element_blank(),
    # plot.title = element_text(vjust = -7, hjust = -0.1),
    # plot.margin = margin(t = -10),
    # panel.grid.minor.y = element_blank()
    axis.title.y = element_text(size = 17),
    axis.title.x = element_text(size = 17)
  )

ggsave("thesis/facilitationconstraint.pdf", height = 6, width = 10)

facilitations %>%
  filter(bucket != 0) %>%
  ggplot(aes(bucket/10, facilitated, color = context_type)) +
  geom_point(size = 3) +
  geom_line(size = 1) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_colour_manual(values = c("#005082", "#f2a365")) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.0)) +
  facet_wrap(~model) +
  labs(
    x = "Binned Constraint score (Probability of most expected word)",
    y = "Proportion of primed instances (F > 0)",
    color = "Scenario"
  ) +
  # ggtitle("(B)") +
  theme_bw(base_size = 20) +
  theme(
    legend.position = "top",
    # plot.title = element_text(vjust = -3, hjust = -0.1),
    # plot.margin = margin(t = -1),
    axis.title.y = element_text(size = 17),
    axis.title.x = element_text(size = 17)
    # panel.grid.minor.x =element_blank()
  )

ggsave("thesis/facilitatedconstraint.pdf", height = 6, width = 10)


# Neutral Table

facilitations %>%
  filter(bucket == 0) %>%
  select(-bucket, -context_type) %>%
  rename(scenario = "dataset")


# Lexical relations 

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
    se = 1.96 * plotrix::std.error(facilitation),
    high = f + se,
    low = f - se,
    facilitated = mean(facilitation > 0)
  ) %>%
  inner_join(relations) %>%
  mutate(
    relation = paste0(relation, " (", n, ")"), 
    relation = factor(relation, levels = rel_levels)
  ) %>%
  ungroup()

relation_facilitation %>%
  filter(bucket != 0) %>%
  ggplot(aes(bucket/10, f, color = context_type, fill = context_type)) +
  geom_point(size = 2) +
  geom_line() +
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.2, color = NA) +
  facet_wrap(~relation, nrow = 2) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("#005082", "#f2a365")) +
  scale_fill_manual(values = c("#005082", "#f2a365")) +
  theme_bw(base_size = 15) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold")
  ) +
  labs(
    x = "Binned Constraint score (Probability of most expected word)",
    y = "Facilitation (in bits)",
    color = "Prime Context",
    fill = "Prime Context"
  )

ggsave("thesis/relationfacilitation.pdf", height = 6, width = 15)

relation_facilitation %>%
  filter(bucket != 0) %>%
  ggplot(aes(bucket/10, facilitated, color = context_type, fill = context_type)) +
  geom_point(size = 2) +
  geom_line() +
  facet_wrap(~relation, nrow = 2) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format()) +
  scale_color_manual(values = c("#005082", "#f2a365")) +
  scale_fill_manual(values = c("#005082", "#f2a365")) +
  theme_bw(base_size = 15) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold")
  ) +
  labs(
    x = "Binned Constraint score (Probability of most expected word)",
    y = "Proportion of primed instances (F > 0)",
    color = "Prime Context",
    fill = "Prime Context"
  )

ggsave("thesis/relationfacilitated.pdf", height = 6, width = 15)

# relation neutral

relation_facilitation %>%
  filter(bucket == 0) %>%
  mutate(
    relation = str_to_title(str_trim(str_replace(relation, "\\((.*?)\\)", ""))),
    se = round(se, 2),
    f = round(f, 2),
    f = paste0(f, "$\\pm$", se),
    facilitated = percent(round(facilitated, 4))
  ) %>%
  select(-high, -low, -bucket, -se) %>%
  pivot_wider(names_from = context_type, values_from = c(f, facilitated)) %>%
  arrange(-n) %>%
  write_csv("thesis/neutral_relation.csv")

# mispriming

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
  geom_point(size = 4) +
  geom_line(size = 1.2) +
  geom_hline(yintercept = 0.5, linetype = 2) +
  annotate("text", x = 0.25, y = 0.25, label = "Majority of Related\nWords are Distractors", color = "black", size = 5) +
  annotate("text", x = 0.80, y = 0.875, label = "Majority of Related\nWords are Primes", color = "black", size = 5) +
  # geom_errorbar(aes(ymin = low_f, ymax = high_f)) +
  # geom_ribbon(aes(fill = model, ymin = low_f, ymax = high_f), alpha = 0.2) +
  # scale_x_continuous(breaks = seq(0, 1, by = .1), limits = c(0, 1.0)) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_colour_manual(values = c("#005082", "#f2a365")) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.0)) +
  guides(
    color = guide_legend(
      title.position = "left",
      # direction = "vertical",
      hjust = -1
    ),
    linetype = guide_legend(
      title.position = "left",
      # direction = "vertical"
    )
  ) +
  facet_wrap(~model) +
  labs(
    x = "Binned Constraint score (Probability of most expected word)",
    y = "Proportion of primed instances",
    color = "Scenario",
    linetype = "Criterion"
  ) +
  theme_bw(base_size = 20) +
  theme(
    legend.position = "top",
    # legend.position = c(0, 1),
    # legend.justification = c(-1, 1),
    axis.text = element_text(size = 17),
    strip.text = element_text(size = 17),
    legend.text = element_text(size = 15),
    legend.title = element_text(size = 15)
    # panel.grid.minor.x =element_blank()
  )

ggsave("thesis/distraction.pdf", height = 7, width = 11)

full_dataset <- read_csv("data/sentences_polysemy_free.csv")

bucket_results %>% 
  filter(category > 7) %>%
  mutate(
    f = -log2(unrelated_probability/related_probability),
    rf = -log2(isolated_probability/related_probability),
  ) %>%
  filter(f < 0) %>%
  select (constraint = category, scenario = context_type, target, related, unrelated, isolated_context, model, f, rf) %>%
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  View()

bucket_results %>% 
  # filter(category > 5) %>%
  mutate(
    f = -log2(unrelated_probability/related_probability),
    rf = -log2(isolated_probability/related_probability),
  ) %>% 
  group_by(category, context_type, model) %>%
  summarize(
    pre = mean(f > 0),
    post = mean(f > 0 & related_probability > isolated_probability),
    post2 = mean(f > 0 & rf > 0)
  )

relation_facilitation_models <- bucket_results %>%
  rename(bucket = "category") %>%
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  # filter(bucket != 0) %>%
  inner_join(relations) %>%
  mutate(
    relation = paste0(relation, " (", n, ")"), 
    relation = factor(relation, levels = rel_levels)
  ) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  select(model, bucket, relation, context_type, target, related, unrelated, facilitation) %>%
  group_by(bucket, context_type, relation, model) %>%
  summarize(
    f = mean(facilitation),
    se = 1.96 * plotrix::std.error(facilitation),
    high = f + se,
    low = f - se,
    facilitated = mean(facilitation > 0)
  ) %>%
  ungroup()

relation_facilitation_models %>%
  filter(bucket != 0, model == "bert-base-uncased") %>%
  ggplot(aes(bucket/10, f, color = context_type, fill = context_type)) +
  geom_point(size = 2) +
  geom_line() +
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.2, color = NA) +
  facet_wrap(~relation, nrow = 2) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("#005082", "#f2a365")) +
  scale_fill_manual(values = c("#005082", "#f2a365")) +
  theme_bw(base_size = 15) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold")
  ) +
  labs(
    x = "Binned Constraint score (Probability of most expected word)",
    y = "Facilitation (in bits)",
    color = "Prime Context",
    fill = "Prime Context"
  )

ggsave("thesis/bertbasefacilitation.pdf", height = 6, width = 15, device = cairo_pdf)

relation_facilitation_models %>%
  filter(bucket != 0, model == "bert-large-uncased") %>%
  ggplot(aes(bucket/10, f, color = context_type, fill = context_type)) +
  geom_point(size = 2) +
  geom_line() +
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.2, color = NA) +
  facet_wrap(~relation, nrow = 2) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("#005082", "#f2a365")) +
  scale_fill_manual(values = c("#005082", "#f2a365")) +
  theme_bw(base_size = 15) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold")
  ) +
  labs(
    x = "Binned Constraint score (Probability of most expected word)",
    y = "Facilitation (in bits)",
    color = "Prime Context",
    fill = "Prime Context"
  )

ggsave("thesis/bertlargefacilitation.pdf", height = 6, width = 15, device = cairo_pdf)

set.seed(1234)

lrts <- bucket_results %>%
  rename(bucket = "category") %>%
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  # filter(bucket != 0) %>%
  # inner_join(relations) %>%
  # mutate(
  #   relation = paste0(relation, " (", n, ")"), 
  #   relation = factor(relation, levels = rel_levels)
  # ) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  # filter(model == "bert-base-uncased", bucket != 0) %>%
  group_by(model, dataset) %>%
  nest() %>%
  mutate(
    lmerfit = map(data, function(x) {
      constraint_model <- lmer(facilitation ~ constraint + (1|target), x, REML = FALSE)
      nonconstraint_model <- lmer(facilitation ~ 1 + (1|target), x, REML = FALSE)
      
      fixed_effect <- coef(summary(constraint_model))[2,1]
      
      tidy(anova(constraint_model, nonconstraint_model, test = "Chisq")) %>%
        filter(term == "constraint_model") %>%
        select(statistic, df, p.value) %>%
        mutate(
          fixed_effect = fixed_effect
        )
    })
  ) %>%
  unnest(lmerfit)

set.seed(1234)

lrt_relations <- bucket_results %>%
  rename(bucket = "category") %>%
  inner_join(
    spp %>%
      select(target, related, unrelated, relation) 
  ) %>%
  # filter(bucket != 0) %>%
  inner_join(relations) %>%
  mutate(
    relation = paste0(relation, " (", n, ")"),
    relation = factor(relation, levels = rel_levels)
  ) %>%
  mutate(
    facilitation = log2(related_probability/unrelated_probability)
  ) %>%
  # filter(model == "bert-base-uncased", bucket != 0) %>%
  group_by(model, dataset, relation, n) %>%
  nest() %>%
  mutate(
    lmerfit = map(data, function(x) {
      constraint_model <- lmer(facilitation ~ constraint + (1|target), x, REML = FALSE)
      nonconstraint_model <- lmer(facilitation ~ 1 + (1|target), x, REML = FALSE)
      
      fixed_effect <- coef(summary(constraint_model))[2,1]
      
      tidy(anova(constraint_model, nonconstraint_model, test = "Chisq")) %>%
        filter(term == "constraint_model") %>%
        select(statistic, df, p.value) %>%
        mutate(
          fixed_effect = fixed_effect
        )
    })
  ) %>%
  unnest(lmerfit) %>%
  select(-data)

lrt_relations %>%
  ungroup() %>%
  mutate(
    relation = str_to_title(str_trim(str_replace(relation, "\\((.*?)\\)", ""))),
    statistic = round(statistic, 2),
    fixed_effect = round(fixed_effect, 2)
  ) %>%
  arrange(model, dataset, -n) %>%
  select(-df, -p.value) %>%
  pivot_wider(names_from = dataset, values_from = c(statistic, fixed_effect)) %>%
  write_csv("thesis/lrt_relations_wider.csv")
