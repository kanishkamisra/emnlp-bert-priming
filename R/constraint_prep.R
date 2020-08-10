library(tidyverse)

contexts <- read_csv("data/results/bert-base-uncased_word_10.csv") %>%
  filter(!target %in% same) %>%
  distinct(isolated_context, constraint, category)

write_csv(contexts, "data/unique_contexts.csv")

