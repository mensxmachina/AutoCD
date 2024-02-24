#install.packages("MXM", repos = "http://cran.us.r-project.org")
library("MXM")

#! /usr/bin/Rscript

cur_path = getwd()
dirname_= dirname(cur_path)
new_path = file.path(dirname_, 'predictive_modeling')
setwd(new_path)


# arguments
args = commandArgs(trailingOnly=TRUE)
dataset_name <- args[1]
target_name <- args[2]
test_name <- args[3]
alpha <- as.double(args[4])
max_k <- strtoi(args[5])



if (length(args)==6) {
  train_idx_name <- args[6]
}



dataset <- read.csv(dataset_name, header=TRUE)
dataset <- as.matrix(sapply(dataset, as.numeric))

if (length(args)==6) {
  train_idx <- read.csv(train_idx_name, header=TRUE)
  r_train_idx <- train_idx$train_idx + 1
  train_data <- dataset[r_train_idx,]
} else{
  train_data <-dataset
}

target_data <- train_data[, target_name]
feature_data <- train_data[, colnames(dataset) != target_name]


# run the SES algorithm
ses_object <- SES(target_data , feature_data, max_k = max_k, threshold = alpha, test=test_name);
selectedVars <- ses_object@selectedVars;
write.csv(selectedVars, 'ses_selectedVars.csv', row.names=FALSE)

