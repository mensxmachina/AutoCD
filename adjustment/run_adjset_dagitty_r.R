library('dagitty')
library("pcalg")


cur_path = getwd()
dirname_= dirname(cur_path)
new_path = file.path(dirname_, 'adjustment')
setwd(new_path)

args = commandArgs(trailingOnly=TRUE)
graph_file_name <- args[1]
graph_type <-args[2]
exposure_file_name <- args[3]
outcome_file_name <- args[4]


graph=as.matrix(read.table(graph_file_name,  header = TRUE, row.names = 1, sep=','))
column_names <- colnames(graph)

exposures=as.matrix(read.table(exposure_file_name,  header = TRUE, row.names = 1, sep=','))
outcomes=as.matrix(read.table(outcome_file_name,  header = TRUE, row.names = 1, sep=','))


g <-pcalg2dagitty(graph, labels=column_names, type=graph_type)


canonical <- dagitty::adjustmentSets(g, exposure = exposures,
                                        outcome = outcomes, 
                                        type = 'canonical', 
                                        effect = "total")

minimal <- dagitty::adjustmentSets(g, exposure = exposures,
                                        outcome = outcomes,
                                        type = 'minimal',
                                        effect = "total")


canonical_res <- unclass(canonical)
minimal_res <- unclass(minimal)

canonical_name = 'canonical_dagitty.csv'
minimal_name = 'minimal_dagitty.csv'


# to be consistent with pcalg csv results
if (length(canonical_res) ==1) {
  if (length(canonical_res[["1"]])==0){
    canonical_res[["1"]]=integer(0)
  }
}

if (length(minimal_res) ==1) {
  if (length(minimal_res[["1"]])==0){
    minimal_res[["1"]]=integer(0)
  }
}

# print(canonical_res[1])
# print(minimal_res[1])

write.csv(canonical_res[1],canonical_name, row.names=FALSE)
write.csv(minimal_res[1], minimal_name, row.names=FALSE)

