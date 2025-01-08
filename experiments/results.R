library(dplyr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
library(readr)

#==> Initialisation  
list_data <- c('Rcatecm','Recm', 'Recmdd', 'Rkmeans', 'Rkmodes', 'Rtskmeans', 'Rsoftecm')
list_model <- c('ECM', 'ECMdd', 'CatECM', 'KMeans', 'KModes', 'TSKmeans', 'Soft-ECM')
list_index <- c('ACC', 'RI', 'NS', 'ASW')


# ==> Fusion
directory <- "outputs/ieee25"
res <- do.call(rbind, lapply(list_data, function(file) {
  file_path <- file.path(directory, paste0(file, ".csv"))
  read.csv(file_path)
}))
write.csv(res, "res.csv")




# ==> SOFTECM
dfN <- read.csv("outputs/Rsoftecm_extracted.csv")
dfN2 <- dfN %>%
  separate(data, into = c("Dataset", "Metric"), sep = "_", fill = "right", remove = FALSE)
dfN2$Metric[dfN2$data %in% c('Abalone','Ecoli','Glass')] <- "Euclidean"
dfN2$Metric[dfN2$data %in% c('BC','Soybean','Lung')] <- "Hamming"
dfN2$Dataset[dfN2$Dataset == "ER"] <- "ERing"
dfN2 <- dfN2 %>% 
  rename(
    Alpha  = alpha,
    Beta   = beta,
    Lambda = lambda)

df <- read.csv("outputs/res.csv", row.names = 1)
dfsoft <- df %>% filter(Algorithmes == "softecm")
df_merged <- dfsoft %>%
  left_join(dfN2, by = c("iter", "Dataset", "Metric", "Alpha", "Beta", "Lambda"))

dfsoft <- df_merged %>%
  group_by(Dataset, Metric, Beta, Lambda) %>%
  summarise(
    ACC = mean(ACC, na.rm = TRUE),
    ARI = mean(ARI, na.rm = TRUE),
    RI  = mean(RI,  na.rm = TRUE),
    NS  = round(mean(N,  na.rm = TRUE),3),
    ASW = mean(ASW, na.rm = TRUE),
    .groups = "drop") %>% 
  mutate(datametric = paste0(Dataset, " (", Metric, ")"))

dfsoft <- dfsoft %>% 
  filter(!datametric %in% c("FM (Euclidean)", "FM (SoftDTW)"))
write.csv(dfsoft, "outputs/parm.csv")




# Extractions tables des resultats
parm <- read.csv("outputs/parm.csv", row.names = 1)
parm <- parm %>% group_by(datametric) %>% filter(NS == min(NS, na.rm = TRUE)) %>% 
  arrange(desc(Beta), Lambda) %>% slice(1)

df <- read.csv("outputs/res.csv", row.names = 1)
res <- df %>%
  mutate(datametric = paste0(Dataset, " (", Metric, ")")) %>%
  rowwise() %>%
  filter(
    (Algorithmes %in% c("kmeans", "kmodes", "tskmeans")) |
      
      (Algorithmes %in% c("catecm", "ecm", "ecmdd") &
         Beta %in% parm$Beta[ parm$datametric == datametric ]) |
      (Algorithmes == "softecm" &
         Beta    %in% parm$Beta[    parm$datametric == datametric ] &
         Lambda  %in% parm$Lambda[  parm$datametric == datametric ])
  ) %>% ungroup()

res_summary <- res %>%
  group_by(Algorithmes, datametric) %>%
  summarise(
    n = n(),
    mean_RI  = mean(RI, na.rm = TRUE),
    sd_RI    = sd(RI, na.rm = TRUE),
    mean_ACC = mean(ACC, na.rm = TRUE),
    sd_ACC   = sd(ACC, na.rm = TRUE),
    .groups = "drop") %>%
  mutate(
    RI  = paste0(round(mean_RI,  3), " (", round(sd_RI,  3), ")"),
    ACC = paste0(round(mean_ACC, 3), " (", round(sd_ACC, 3), ")"))
write.csv(res_summary, "outputs/res_summary.csv", row.names = FALSE)

res_summary<- read.csv("outputs/res_summary.csv")
print(res_summary, n = Inf)






