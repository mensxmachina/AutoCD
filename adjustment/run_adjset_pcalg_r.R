library('pcalg')

cur_path = getwd()
dirname_= dirname(cur_path)
new_path = file.path(dirname_, 'adjustment')
setwd(new_path)

args = commandArgs(trailingOnly=TRUE)
graph_name <- args[1]
graph_type <-args[2]
x_idx <- args[3]
y_idx <- args[4]

x_idx=as.numeric(x_idx)
y_idx=as.numeric(y_idx)
x_idx =x_idx+1
y_idx = y_idx+1


graph=as.matrix(read.table(graph_name,  header = TRUE, row.names = 1, sep=','))

canonical = adjustment(amat = graph, amat.type = graph_type, x = x_idx, y = y_idx, set.type ="canonical")
minimal = adjustment(amat = graph, amat.type = graph_type, x = x_idx, y = y_idx, set.type ="minimal")

canonical_name = 'canonical_pcalg.csv'
minimal_name = 'minimal_pcalg.csv'
write.csv(canonical,canonical_name, row.names=FALSE)
write.csv(minimal, minimal_name, row.names=FALSE)

