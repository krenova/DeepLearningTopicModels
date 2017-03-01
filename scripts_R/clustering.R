
setwd('/home/ekhongl/Codes/DL - Topic Modelling')
library(ggplot2)
library(dplyr)

map_class <- function(x){
  return( c(3,0,0,0,0,0,5,1,1,1,1,2,2,2,2,3,4,4,4,3)[0:19 %in% x] )
}



dat <- read.csv('data/ae_features_2000_nonoise.csv',check.names=FALSE)


dat_raw <- read.table('data/raw_20news/20news.csv',header = T, sep=',',
					row.names = NULL, stringsAsFactors = FALSE)


X <- dat[,2:ncol(dat)]
Y <- as.factor(dat[,1])
Y_overview <- sapply(Y,map_class)

for (i in 1:ncol(X)) {
	hist(X[,i],breaks=100)
	invisible(readline(prompt="Press [enter] to continue"))
}

hist(unlist(X),breaks=50)




#----------------------------------------------------------------
X.clust <- kmeans(X,centers = 20, nstart=2000, trace =1)
X.clust_overview <- kmeans(X,centers = 6, nstart=2000, trace =1)

for (i in 1:20) {
	plt_dat <- data.frame(labels = Y[X.clust$cluster == i]) %>%
					group_by(labels) %>%
					summarize( freq = n())
	plt <- ggplot(plt_dat, aes(x=labels, y = freq)) + geom_bar(stat="identity")
	print(plt)
	invisible(readline(prompt="Press [enter] to continue"))
}

for (i in sort(unique(X.clust_overview$cluster))) {
    print( paste("----- Current cluster: ", i, " -----"))
	plt_dat <- data.frame(labels = Y_overview[X.clust_overview$cluster == i]) %>%
					group_by(labels) %>%
					summarize( freq = n())
	plt <- ggplot(plt_dat, aes(x=labels, y = freq)) + geom_bar(stat="identity")
	print(plt)
	invisible(readline(prompt="Press [enter] to continue"))
}

table(X.clust$cluster)
table(Y)
table(X.clust_overview$cluster)
table(Y_overview)

write.csv(X.clust$cluster, '../data/clustered_output_nonoise.csv')
write.csv(X.clust_overview$cluster, '../data/clustered_overview_output_nonoise.csv')
#----------------------------------------------------------------




#----------------------------------------------------------------
library(mixtools)
X_bin <- (X >0.1)*1

mixout <- multmixEM(X_bin, lambda = NULL, theta = NULL, k = 6,
					maxit = 10000, epsilon = 1e-08, verb = TRUE)

mixout.clust <- apply( mixout$posterior, 1, function(x) which(x == max(x)) )


for (i in sort(unique(mixout.clust))) {
    print( paste("----- Current cluster: ", i, " -----"))
	plt_dat <- data.frame(labels = Y_overview[mixout.clust == i]) %>%
					group_by(labels) %>%
					summarize( freq = n())
	plt <- ggplot(plt_dat, aes(x=labels, y = freq)) + geom_bar(stat="identity")
	print(plt)
	invisible(readline(prompt="Press [enter] to continue"))
}
#----------------------------------------------------------------




#----------------------------------------------------------------
library(e1071)
# import your binary data with read.table or read.delim; the following
# example uses random data
y <- matrix(sample(c(0,1), 100, replace=TRUE), 10, 10,
dimnames=list(paste("g", 1:10, sep=""), paste("t", 1:10, sep="")))
disma <- hamming.distance(X_bin)
hr <- hclust(as.dist(disma))
plot(as.dendrogram(hr), edgePar=list(col=3, lwd=4), horiz=T)

hr.clust <- cutree(hr,k=6)

for (i in sort(unique(hr.clust))) {
    print( paste("----- Current cluster: ", i, " -----"))
	plt_dat <- data.frame(labels = Y_overview[hr.clust == i]) %>%
					group_by(labels) %>%
					summarize( freq = n())
	plt <- ggplot(plt_dat, aes(x=labels, y = freq)) + geom_bar(stat="identity")
	print(plt)
	invisible(readline(prompt="Press [enter] to continue"))
}
#----------------------------------------------------------------


	
	
