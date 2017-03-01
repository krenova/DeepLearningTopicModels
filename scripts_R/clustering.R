
setwd('/home/ekhongl/Codes/DL - Topic Modelling')
library(ggplot2)
library(dplyr)






#----------------------------------------------------------------
# [0] Data Prep
#----------------------------------------------------------------
map_class <- function(x){
  return( c(3,0,0,0,0,0,5,1,1,1,1,2,2,2,2,3,4,4,4,3)[0:19 %in% x] )
}

Y_levels <- c('alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos','rec.motorcycles',
                  'rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med',
                  'sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast',
                  'talk.politics.misc','talk.religion.misc')

Y_overview_levels <- c('computer','hobby','science','religion','politics','sales')

dat <- read.csv('data/ae_features_2000_nonoise.csv',check.names=FALSE)

dat_raw <- read.table('data/raw_20news/20news.csv',header = T, sep=',',
					row.names = NULL, stringsAsFactors = FALSE)


X <- dat[,2:ncol(dat)]
Y <- as.factor(dat[,1])
Y_overview <- as.factor(sapply(Y,map_class))
levels(Y) <- Y_levels
levels(Y_overview) <- Y_overview_levels
#----------------------------------------------------------------



#----------------------------------------------------------------
# [1] Viewing the distribution of each neuron's output
#----------------------------------------------------------------
for (i in 1:ncol(X)) {
	hist(X[,i],breaks=100)
	invisible(readline(prompt="Press [enter] to continue"))
}

hist(unlist(X),breaks=50)
#----------------------------------------------------------------




#----------------------------------------------------------------
# [2] K-means clustering to segment the data
#----------------------------------------------------------------
X.clust <- kmeans(X,centers = 20, nstart=2000, trace =1)
X.clust_overview <- kmeans(X,centers = 6, nstart=2000, trace =1)

for (i in 1:20) {
	plt_dat <- data.frame(labels = Y[X.clust$cluster == i]) %>%
					group_by(labels) %>%
					summarize( freq = n())
	plt <- ggplot(plt_dat, aes(x=labels, y = freq)) + 
	            geom_bar(stat="identity") + 
	            theme(axis.text.x = element_text(angle = 60, hjust = 1))
	print(plt)
	invisible(readline(prompt="Press [enter] to continue"))
	rm(plt)
}

for (i in sort(unique(X.clust_overview$cluster))) {
    print( paste("----- Current cluster: ", i, " -----"))
	plt_dat <- data.frame(labels = Y_overview[X.clust_overview$cluster == i]) %>%
					group_by(labels) %>%
					summarize( freq = n())
	plt <- ggplot(plt_dat, aes(x=labels, y = freq)) + geom_bar(stat="identity")
	print(plt)
	invisible(readline(prompt="Press [enter] to continue"))
	rm(plt)
}

table(Y,X.clust$cluster)
table(Y_overview,X.clust_overview$cluster)

write.csv(X.clust$cluster, 'data/clustered_output_nonoise.csv')
write.csv(X.clust_overview$cluster, 'data/clustered_overview_output_nonoise.csv')
#----------------------------------------------------------------


save(Y,Y_overview, X.clust, X.clust_overview, file = "data/cluster_results.RData")


# #----------------------------------------------------------------
# # [3] multinomial mixture modelling to segment the data
# #----------------------------------------------------------------
# library(mixtools)
# X_bin <- (X >0.1)*1
# 
# mixout <- multmixEM(X_bin, lambda = NULL, theta = NULL, k = 6,
# 					maxit = 10000, epsilon = 1e-08, verb = TRUE)
# 
# mixout.clust <- apply( mixout$posterior, 1, function(x) which(x == max(x)) )
# 
# 
# for (i in sort(unique(mixout.clust))) {
#     print( paste("----- Current cluster: ", i, " -----"))
# 	plt_dat <- data.frame(labels = Y_overview[mixout.clust == i]) %>%
# 					group_by(labels) %>%
# 					summarize( freq = n())
# 	plt <- ggplot(plt_dat, aes(x=labels, y = freq)) + geom_bar(stat="identity")
# 	print(plt)
# 	invisible(readline(prompt="Press [enter] to continue"))
# 	rm(plt)
# }
# #----------------------------------------------------------------
# 
# 
# 
# 
# #----------------------------------------------------------------
# # [4] hiearchical clustering using hamming distance to segment the data
# #----------------------------------------------------------------
# library(e1071)
# # import your binary data with read.table or read.delim; the following
# # example uses random data
# y <- matrix(sample(c(0,1), 100, replace=TRUE), 10, 10,
# dimnames=list(paste("g", 1:10, sep=""), paste("t", 1:10, sep="")))
# disma <- hamming.distance(X_bin)
# hr <- hclust(as.dist(disma))
# plot(as.dendrogram(hr), edgePar=list(col=3, lwd=4), horiz=T)
# 
# hr.clust <- cutree(hr,k=6)
# 
# for (i in sort(unique(hr.clust))) {
#     print( paste("----- Current cluster: ", i, " -----"))
# 	plt_dat <- data.frame(labels = Y_overview[hr.clust == i]) %>%
# 					group_by(labels) %>%
# 					summarize( freq = n())
# 	plt <- ggplot(plt_dat, aes(x=labels, y = freq)) + geom_bar(stat="identity")
# 	print(plt)
# 	invisible(readline(prompt="Press [enter] to continue"))
# }
# #----------------------------------------------------------------


	
	
