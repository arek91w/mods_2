library(data.tree)
library(pROC)
library(rpart)
library(e1071)
library(nnet)
library(FNN)
library(ModelMetrics)
library(cluster)


##################################KNN###########################################
MinMax <- function(x, new_min, new_max){
  return(((x - min(x)) / (max(x) - min(x))) * (new_max - new_min) + new_min)
}

MinMax_pred <- function(x, new_min, new_max, old_min, old_max){
  return((( x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min)
}

d_euklides <- function(x_i, x_j){
  return(sqrt(sum((x_i - x_j)^2)))
}

d_hamming <- function(x_i, x_j){
  return(sum(x_i != x_j)/length(x_i))
}

d_interwal <- function(x_i, x_j, x_lvl){
  return(sum(abs(as.numeric(x_i)-as.numeric(x_j))/(x_lvl - 1)))
}

d_gower <- function(x_i, x_j, x_class, x_min, x_max){
  n <- length(x_class)
  sum <- 0
  for (i in 1:n){
    if(is.numeric(x_class[i])){
      g <- abs(x_i[i] - x_j[i])/(x_max[i] - x_min[i])
      sum <- sum + g
    }
    else if(is.factor(x_class[i])){ 
      g <- as.double(x_i[i] != x_j[i])
      sum <- sum + g
    }
    else{
      z_min <- (x_min[i]-1)/(x_max[i]-1)
      z_max <- 1
      z_i <- (as.numeric(x_i[i])-1)/(x_max[i]-1)
      z_n <- (as.numeric(x_j[i])-1)/(x_max[i]-1)
      g <- abs(z_i - z_n)/(z_max - z_min)
      sum <- sum+g
    }
  }
  wynik <- sum/n
  return(wynik[[1]])
}


KNNtrain <- function(X, y_tar, k, XminNew, XmaxNew){
  if (!all(!is.na(X)) | !all(!is.na(y_tar))){
    print("Wystepuja braki w danych")
  }
  else if (k%%1!=0 | k < 0){
    print("Nieprawidlowa wartosc k")
  }
  else if (!is.matrix(X) & !is.data.frame(X)){
    print("Niewlasciwy typ X")
  }
  else{
    n <- ncol(X)
    min_org <- vector(length=n)
    max_org <- vector(length=n)
    minmax_new <- c(XminNew, XmaxNew)
    for (i in 1:n){
      if(is.numeric(X[,i]) | is.ordered(X[,i])){
        min_org[i] <- min(X[,i])
        max_org[i] <- max(X[,i])
      }
      else{
        min_org[i] <- NA
        max_org[i] <- NA
      }
      if(is.numeric(X[,i])){
        X[,i] <- MinMax(X[,i], XminNew, XmaxNew)
      }
    }
    attr(X, "minOrg") <- min_org
    attr(X, "maxOrg") <- max_org
    attr(X, "minmaxNew") <- minmax_new 
    knn <- list()
    knn[["X"]] <- X
    knn[["y"]] <- y_tar
    knn[["k"]] <- k
    return(knn)
  }
}

KNNpred <- function(KNNmodel, X){
  
  if(!all(attr(KNNmodel$X, "names")==attr(X, "names"))){
    print("Niezgodne kolumny")
  }
  else if(!all(!is.na(X))){
    print("Braki danych w zbiorze testowym")
  }
  
  n <- ncol(X)
  for (i in 1:n){
    if(is.numeric(X[,i])){
      X[,i] <- MinMax_pred(X[,i],
                           attr(KNNmodel$X, "minmaxNew")[1],
                           attr(KNNmodel$X, "minmaxNew")[2],
                           attr(KNNmodel$X, "minOrg")[i],
                           attr(KNNmodel$X, "maxOrg")[i])
    }
  }
  n_train <- nrow(KNNmodel$X)
  n_pred <- nrow(X)
  dist <- matrix(0, n_train, n_pred)
  
  if (all(sapply(X, is.numeric))){ 
    for (i in 1:n_train){
      for (j in 1:n_pred){
        dist[i,j] <- d_euklides(KNNmodel$X[i,], X[j,])
      }
    }
  }
  else if (all(sapply(X, is.character) | sapply(X, is.factor))){
    for (i in 1:n_train){
      for (j in 1:n_pred){
        dist[i,j] <- d_hamming(KNNmodel$X[i,], X[j,])
      }
    }
  }
  else if (all(sapply(X, is.ordered))){
    lvl <- vector(length=n)
    for (i in 1:n){
      lvl[i]<-length(levels(KNNmodel$X[,i]))
    }
    for (i in 1:n_train){
      for (j in 1:n_pred){
        dist[i,j] <- d_interwal(KNNmodel$X[i,], X[j,], lvl)
      }
    }
  }
  else{
    xmin <- attr(KNNmodel$X, "minOrg")
    xmax <- attr(KNNmodel$X, "maxOrg")
    
    for (i in 1:n_train){
      for (j in 1:n_pred){
        dist[i,j] <- d_gower(KNNmodel$X[i,], X[j,], classList, xmin, xmax)
      }
    }
  }
  if (is.numeric(KNNmodel$y)){
    pred <- double(n_pred)
    
    for(i in 1:n_pred){
      kNaj <- order(dist[,i])
      kNaj <- kNaj[1:KNNmodel$k]
      y_hat <- mean(KNNmodel$y[kNaj])
      pred[i] <- y_hat
    }
  }
  else {
    pred <- data.frame()
    
    for(i in 1:n_pred){
      kNaj <- order(dist[,i])
      kNaj <- kNaj[1:KNNmodel$k]
      y_hat <- KNNmodel$y[kNaj]
      tmp <- data.frame(row.names = 1)
      
      for(j in unique(y_hat)) {
        tmp[[j]] <- sum(y_hat == j)/KNNmodel$k
      }
      pred <- dplyr::bind_rows(pred, tmp)
    }
    pred[is.na(pred)] = 0
    row.names(pred) <- NULL
    pred[['klasa']] <- colnames(pred)[max.col(pred, ties.method="first")]
  }
  return(pred)
}

###########################TREE#################################################
StopIfNot <- function(Y, X, data, type, depth, minobs, overfit, cf){ #
  if (!is.data.frame(data)){
    print("Obiekt dane nie jest to ramka danych")
    return(FALSE)
  }
  else if (!all(X %in% colnames(data))){
    print("X niezgodne z obiektem dane")
    return(FALSE)
  }
  else if (!Y %in% colnames(data)){
    print("Y niezgodne z obiektem dane")
    return(FALSE)
  }
  else if ( !all(!is.na(data[,X]))){
    print("Wystepuja braki danych w X")
    return(FALSE)
  }
  else if (!all(!is.na(data[,Y]))){
    print("Wystepuja braki danych w Y")
    return(FALSE)
  }
  else if (depth <= 0){
    print("depth <= 0")
    return(FALSE)
  }
  else if (minobs <= 0){
    print("minobs <= 0")
    return(FALSE)
  }
  else if (!type %in% c("Gini", "Entropy", "SS")){
    print("Nieprawidlowa wartosc type")
    return(FALSE)
  }
  else if (!overfit %in% c("none", "prune")){
    print("Nieprawidlowa wartosc overfit")
    return(FALSE)
  }
  else if (cf <= 0 | cf > 0.5){
    print("Nieprawidlowa wartosc cf")
    return(FALSE)
  }
  else if (is.factor(data[,Y]) & type=="SS"){
    print("Nieprawidlowy type (SS) dla klasyfikacji")
    return(FALSE)
  }
  else if (is.numeric(data[,Y]) & type=="Gini"){
    print("Nieprawidlowy type (Gini) dla regresji")
    return(FALSE)
  }
  else if (is.numeric(data[,Y]) & type=="Entropy"){
    print("Nieprawidlowy type (Entropy) dla regresji")
    return(FALSE)
  }
  else{
    return(TRUE)
  }
}

Entropy <- function(prob){
  res <- prob * log2(prob)
  res[ prob == 0 ] <- 0
  res <- -sum(res)
  return(res)
}

Gini <- function(prob){
  res <- prob**2
  res <- 1-sum(res)
  return(res)
}

SS <- function(y_val){
  res <- (y_val - mean(y_val))**2
  res <- sum(res)
  return(res)
}


Prob <- function(y){
  res <- unname(table(y))
  res <- res/sum(res)
  return(res)
}

AssignInitialMeasures <- function(tree, Y, data, type, depth) {
  tree$Depth <- 0
  if (type == 'Gini') {
    val <- Gini(Prob(data[,Y])) 
  }
  else if (type == 'Entropy') {
    val <- Entropy(Prob(data[,Y]))
  }
  else {
    val <- SS(data[,Y])
  }
  tree$Val <- val
  return(tree)
}

AssignInfo <- function(tree, Y, X, data, type, depth, minobs, overfit, cf) {
  tree$Y <- Y
  tree$X <- X
  tree$data <- data
  tree$type <- type
  tree$depth <- depth
  tree$minobs <- minobs
  tree$overfit <- overfit
  tree$cf <- cf
  return(tree)
}

SplitNum <- function(Y, x, parentVal, splits, minobs, type){
  
  n <- length(x)
  res <- data.frame( matrix( 0, length(splits), 6 ) )
  colnames( res ) <- c("InfGain","lVal","rVal","point","ln","rn")
  
  
  for( i in 1:length(splits) ){
    
    partition <- x <= splits[i]
    ln <- sum( partition )
    rn <- n - ln
    
    if( any( c(ln,rn) < minobs  | is.na(ln) | is.na(rn)) ){
      res[i,] <- 0
    }else{
      if(type == 'Entropy') {
        lVal <- Entropy( Prob( Y[partition] ) )
        rVal <- Entropy( Prob( Y[!partition] ) )
      }
      else if(type == 'Gini') {
        lVal <- Gini( Prob( Y[partition] ) )
        rVal <- Gini( Prob( Y[!partition] ) )
      }
      else if (type == 'SS') {
        lVal <- SS(Y[partition])
        rVal <- SS(Y[!partition])
      }
      else {
        stop()
      }
      
      InfGain <- parentVal - ( lVal * ln/n  + rVal * rn/n )
      
      res[i,"InfGain"] <- InfGain
      res[i,"lVal"] <- lVal
      res[i,"rVal"] <- rVal
      res[i,"point"] <- splits[i]
      res[i,"ln"] <- ln
      res[i,"rn"] <- rn
    }
  }
  
  incl <- res$ln >= minobs & res$rn >= minobs & res$InfGain > 0
  res <- res[ incl, , drop = F ]
  best <- which.max( res$InfGain )
  res <- res[ best, , drop = F ]
  
  return(res)
}

SplitVar <- function( Y, x, parentVal, minobs , type){
  
  s <- unique( x )
  if( length(x) == 1 ){
    splits <- s
  }
  else {
    splits <- head( sort( s ), -1 )
  }
  if (type == 'Gini') {
    res <- SplitNum(Y, x, parentVal, splits, minobs, 'Gini')
  }   
  else if (type == 'Entropy') {
    res <- SplitNum(Y, x, parentVal, splits, minobs, 'Entropy')
  } 
  else if (type == 'SS') {
    res <- SplitNum(Y, x, parentVal, splits, minobs, 'SS')
  }
  else {
    stop()
  }
  
  return(res)
}


FindBestSplit <- function(Y, X, data, parentVal, type, minobs) {
  
  if(type == 'Entropy') {
    
    res <- sapply(X, function(i){
      SplitVar(data[,Y], data[,i], parentVal, minobs, 'Entropy')
    }, simplify = F )
    
  } 
  else if(type == 'Gini') {
    
    res <- sapply(X, function(i){
      SplitVar(data[,Y], data[,i], parentVal, minobs, 'Gini')
    }, simplify = F )
    
  } 
  else if(type == 'SS') {
    
    res <- sapply(X, function(i){
      SplitVar(data[,Y], data[,i], parentVal, minobs, 'SS')
    }, simplify = F )
    
  }
  else {
    stop()
  }
  
  res <- do.call( "rbind", res )
  best <- which.max( res$InfGain )
  res <- res[ best, , drop = F ]
  
  return(res)
}

BuildTree <- function( node, Y, Xnames, data, type, depth, minobs ){
  node$Count <- nrow( data )
  
  if (is.factor(data[,Y])) {
    node$Prob <- Prob( data[,Y] )
    node$Class <- levels( data[,Y] )[ which.max(node$Prob) ]}
  if (is.numeric(data[,Y])) {
    node$Class <- mean(data[,Y])
  }
  
  bestSplit <- FindBestSplit(Y, Xnames, data, node$Val, type, minobs)
  
  ifStop <- nrow(bestSplit) == 0 
  
  if( node$Depth == depth | ifStop | (all(node$Prob %in% c(0,1)) & is.factor(data[,Y]))){
    node$Leaf <- "*"
    return( node )
  }
  
  if( (node$Depth == depth | ifStop) & !is.factor(data[,Y])){
    node$Leaf <- "*"
    return( node )
  }
  
  splitIndx <- data[, rownames(bestSplit) ] <= bestSplit$point
  childFrame <- split(data, splitIndx)
  namel <- sprintf("%s <= %s", rownames(bestSplit), bestSplit$point)
  childL <- node$AddChild(namel)
  childL$Depth <- node$Depth + 1
  childL$Val <- bestSplit$lVal
  
  BuildTree(childL, Y, Xnames, childFrame[["TRUE"]], type, depth, minobs)
  
  namer <- sprintf("%s >  %s",  rownames(bestSplit), bestSplit$point)
  childR <- node$AddChild(namer)
  childR$Depth <- node$Depth + 1
  childR$Val <- bestSplit$rVal
  
  BuildTree(childR, Y, Xnames, childFrame[["FALSE"]], type, depth, minobs)
}

PruneTree<-function(){}

Tree <- function(Y, X, data, type, depth, minobs, overfit, cf) {
  StopIfNot(Y, X, data, type, depth, minobs, overfit, cf)
  
  tree <- Node$new("Root")
  tree$Count <- nrow(data)
  
  AssignInitialMeasures(tree, Y, data, type, depth)
  PruneTree()
  BuildTree(tree, Y, X, data, type, depth, minobs)
  AssignInfo(tree, Y, X, data, type, depth, minobs, overfit, cf)
  
  return(tree)
}

PredictTreeHelp <- function(tree, cols, values) {
  if (tree$isLeaf) return (tree$Class)
  
  for (i in 1:length(cols)) {
    assign(cols[i], values[i])
  }
  
  for (j in attributes(tree$children)$names) {
    if (eval(parse(text = j))) {
      child <- tree$children[[j]]
      child$Class <- tree$children[[j]]$Class
    }
    
  }
  return(PredictTreeHelp(child, cols, values))
}

PredTree <- function(df, tree, num=T) {
  pred <- c()
  for (i in 1:nrow(df)) {
    p <- PredictTreeHelp(tree, colnames(df), df[i,])
    if (num==T) {
      p <- as.numeric(p)
    }
    pred <- append(pred, p)
  }
  return(pred)
}

##############################NN################################################
MinMax_nn <- function( x ){
  return( ( x - min(x) ) / ( max(x) - min(x) ) )
}  
MinMaxOdwrot_nn <- function( x, y_min, y_max ){
  return(  x * (y_max - y_min) + y_min )
} 

sigmoid <- function( x ){
  return( 1 / (1 + exp( -x ) ) )
}
dsigmoid <- function( x ){
  return( x * (1 - x) )
}
ReLu <- function( x ){
  return( ifelse( x <= 0, 0, x ) )
}
dReLu <- function( x ){
  return( ifelse( x <= 0, 0, 1 ) )
}
lossSS <- function( y_tar, y_hat ){
  return( 1/2 * sum( ( y_tar - y_hat )^2 ) )
}
SoftMax <- function(x){
  exp( x ) / sum( exp( x ) )
}

Wprzod <-function(X, W1, W2, W3, nn_type){
  if (nn_type %in% c('reg', 'bin', 'mclass')){
    if (nn_type == 'bin'){
      h1 <- cbind(matrix( 1, nrow = nrow(X)), sigmoid(X %*% W1))
      h2 <- cbind(matrix( 1, nrow = nrow(X)), sigmoid(h1 %*% W2))
      y_hat <- sigmoid(h2 %*% W3)
    }
    else if (nn_type == 'mclass'){
      h1 <- cbind( matrix(1, nrow = nrow(X)), sigmoid(X %*% W1))
      h2 <- cbind( matrix(1, nrow = nrow(X) ), sigmoid( h1 %*% W2))
      y_hat <- matrix(t(apply(h2 %*% W3, 1, SoftMax)), nrow = nrow(X))
    }
    else{
      h1 <- cbind(matrix(1, nrow = nrow(X)), ReLu(X %*% W1 ))
      h2 <- cbind(matrix(1, nrow = nrow(X)), ReLu(h1 %*% W2))
      y_hat <- h2 %*% W3 
    }
    return(list(y_hat=y_hat, H1=h1, H2=h2))
  }
  else{
    stop()
  } 
}

Wstecz <-function(X, y_tar, y_hat, W1, W2, W3, H1, H2, lr, nn_type){
  if (nn_type %in% c('reg', 'bin', 'mclass')){
    if (nn_type == 'bin'){
      dy_hat <- (y_tar - y_hat) * dsigmoid(y_hat)
      dW3 <- t(H2) %*% dy_hat
      dH2<- dy_hat %*% t(W3) * dsigmoid(H2)
      dW2 <- t(H1) %*% dH2[,-1]
      dH1<- dH2[,-1] %*% t(W2) * dsigmoid(H1)
      dW1 <- t(X) %*% dH1[,-1]
    }
    else if (nn_type == 'mclass'){
      dy_hat <- (y_tar - y_hat) / nrow(X)
      dW3 <- t(H2) %*% dy_hat
      dH2<- dy_hat %*% t(W3) * dsigmoid(H2)
      dW2 <- t(H1) %*% dH2[,-1]
      dH1<- dH2[,-1] %*% t(W2) * dsigmoid(H1)
      dW1 <- t(X) %*% dH1[,-1]
    }
    else{
      dy_hat <- (y_tar - y_hat)
      dW3 <- t(H2) %*% dy_hat
      dH2<- dy_hat %*% t(W3) * dReLu(H2)
      dW2 <- t(H1) %*% dH2[,-1]
      dH1<- dH2[,-1] %*% t(W2) * dReLu(H1)
      dW1 <- t(X) %*% dH1[,-1]
    }
    W1 <- W1 + lr * dW1
    W2 <- W2 + lr * dW2
    W3 <- W3 + lr * dW3
    return(list(W1=W1, W2=W2, W3=W3))
  }
  else{
    stop()
  } 
}

TrainNN <- function(x, y_tar, nn_type, h = c(5,5), lr = 0.01, iter = 10000, seed = 666){
  if (nn_type %in% c('reg', 'bin', 'mclass')){
    set.seed(seed)
    X <- cbind(rep(1, nrow(x) ), x)
    W1 <- matrix(runif(ncol(X) * h[1], -1, 1), nrow = ncol(X))
    W2 <- matrix(runif((h[1]+1) * h[2], -1, 1), nrow = h[1] + 1)
    W3 <- matrix(runif((h[2]+1) * ncol(y_tar), -1, 1), nrow = h[2] + 1)
    for(i in 1:iter){
      sygnalwprzod <- Wprzod(X, W1, W2, W3, nn_type)
      sygnalwtyl <- Wstecz(X, y_tar, y_hat = sygnalwprzod$y_hat, W1, W2, W3, H1 = sygnalwprzod$H1, H2 = sygnalwprzod$H2, lr, nn_type)
      W1 <- sygnalwtyl$W1
      W2 <- sygnalwtyl$W2
      W3 <- sygnalwtyl$W3
    }
    return(list(y_hat = sygnalwprzod$y_hat, W1 = W1, W2 = W2, W3 = W3))
  }
  else{
    stop()
  }
}

PredNN <- function(xnew, nn, nn_type){
  if (nn_type %in% c('reg', 'bin', 'mclass')){
    xnew <- cbind(rep(1, nrow(xnew)), xnew)
    h1 <- cbind(matrix(1, nrow = nrow(xnew)), sigmoid(xnew %*% nn$W1))
    h2 <- cbind(matrix(1, nrow = nrow(xnew)), sigmoid(h1 %*% nn$W2))
    
    if (nn_type == 'reg'){
      y_hat <- ReLu(h2 %*% nn$W3)
    }
    else{
      y_hat <- sigmoid(h2 %*% nn$W3)
    }
    return(y_hat)
  }
  else{
    stop()
  }
}

##############################EVALUATION########################################
MAE <- function(y_tar, y_hat){
  return(mean(abs(y_tar - y_hat)))
}

MSE <- function(y_tar, y_hat){
  return(mean((y_tar-y_hat)^2))
}

MAPE <- function(y_tar, y_hat){
  return(mean(abs((y_tar - y_hat)/y_tar)))
}

AUC <- function( y_tar, y_hat ){
  
  y_tar <- y_tar[order(-y_hat,y_tar)]
  n_obs <- length(y_tar)
  auc_mat <- as.data.frame(matrix(0, n_obs, 9, dimnames = list(NULL,c("target", "index", "TP", "FP", "TN", "FN", "sens", "1-spec", "area"))))
  
  auc_mat[,'target'] <- y_tar
  auc_mat[,'index'] <- 1:n_obs
  auc_mat[,'TP'] <- cumsum(auc_mat[,'target'])
  auc_mat[,'FP'] <- cumsum(!auc_mat[,'target'])
  auc_mat[,'TN'] <- n_obs-sum(auc_mat[,'target'])-auc_mat[,'FP']
  auc_mat[,'FN'] <- n_obs-auc_mat[,'TP']-auc_mat[,'FP']-auc_mat[,'TN']
  auc_mat[,'sens'] <- auc_mat[,'TP']/(auc_mat[,'TP']+auc_mat[,'FN'])
  auc_mat[,'1-spec'] <- 1-(auc_mat[,'TN']/(auc_mat[,'FP']+auc_mat[,'TN']))
  auc_mat[1,'area'] <- 0
  auc_mat[2:n_obs,'area'] <- (auc_mat[2:n_obs,'sens']+auc_mat[auc_mat[2:n_obs,'index']-1,'sens'])*(auc_mat[2:n_obs,'1-spec']-auc_mat[auc_mat[2:n_obs,'index']-1,'1-spec'])
  
  auc <- sum(auc_mat[,'area'])*(1/2)
  
  return(auc)
}

youden_index <- function(y_tar,y_hat) {
  
  j_list <- c()
  for(i in seq(0, 1, 0.01)){
    coef_mat <- table(y_tar, y_hat = ifelse(y_hat <= i, 0, 1))
    
    if(nrow(coef_mat) == 2 & ncol(coef_mat) == 2) {
      sens <- coef_mat[2,2] / (coef_mat[2,2] + coef_mat[2,1])
      spec <- coef_mat[1,1] / (coef_mat[1,1] + coef_mat[1,2])
      j <- sens + spec - 1
      j_list <- append(j_list, j)
    }
  }
  return(max(j_list))
}

confusion_matrix <- function(y_tar,y_hat) {
  
  j <- youden_index(y_tar,y_hat)
  conf_mat = table(y_tar, y_hat = factor(ifelse(y_hat <= j, 0, 1), levels=0:1))
  return(conf_mat)
}

sensitivity <- function(conf_mat) {
  sens = conf_mat[2,2] / (conf_mat[2,2] + conf_mat[2,1])
  return(sens)
}

specificity <- function(conf_mat) {
  spec = conf_mat[1,1] / (conf_mat[1,1] + conf_mat[1,2])
  return(spec)
}

accuracy <- function(conf_mat) {
  acc = (conf_mat[1,1] + conf_mat[2,2]) / (sum(conf_mat))
  return(acc)
}

ModelOcena <- function(y_tar, y_hat){
  
  if (is.factor(y_tar) & is.numeric(y_hat)){
    y_tar <- as.numeric(as.character(y_tar))
    AUC <- AUC(y_tar, y_hat)
    
    conf_mat <- confusion_matrix(y_tar, y_hat)

    j <- youden_index(y_tar,y_hat)
    
    sens <- sensitivity(conf_mat)
    spec <- specificity(conf_mat)
    acc <- accuracy(conf_mat)
    
    scores <- c(AUC, sens, spec, acc)
    names(scores) <- c('AUC', 'Czulosc', 'Specyficznosc', 'Jakosc')
    
    results <- list('Mat' = conf_mat, 'J' = j, 'Miary' = scores)
    
    return(results)
    
  }
  else if ((is.factor(y_tar) & is.character(y_hat)) | (is.factor(y_tar) & is.factor(y_hat))){
    results <- mean(y_tar == y_hat)
    
  }
  else if (is.numeric(y_tar) & is.numeric(y_hat)){
    results <- vector(length = 3)
    names(results) <- c("MAE", "MSE", "MAPE")
    results["MAE"] <- MAE(y_tar, y_hat)
    results["MSE"] <- MSE(y_tar, y_hat)
    results["MAPE"] <- MAPE(y_tar, y_hat)
    
    return(results)
  }
  else {
    print("Nieprawidlowy typ danych wejsciowych.")
  }
}


MeanEvaluation <- function(result_table, parTune){
  
  if ('AUCT' %in% colnames(result_table)){
    result_table_avg <- data.frame(kFold_mat=1:nrow(parTune), parTune, AUCT=0, CzuloscT=0, SpecyficznoscT=0, JakoscT=0, AUCW=0, CzuloscW=0, SpecyficznoscW=0, JakoscW=0)
    for (i in 1:nrow(parTune)){
      result_table_avg[i, "AUCT"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"AUCT"])
      result_table_avg[i, "CzuloscT"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"CzuloscT"])
      result_table_avg[i, "SpecyficznoscT"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"SpecyficznoscT"])
      result_table_avg[i, "JakoscT"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"JakoscT"])
      result_table_avg[i, "AUCW"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"AUCW"])
      result_table_avg[i, "CzuloscW"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"CzuloscW"])
      result_table_avg[i, "SpecyficznoscW"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"SpecyficznoscW"])
      result_table_avg[i, "JakoscW"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"JakoscW"])
    }
  }
  else if (('JakoscT' %in% colnames(result_table))){
    result_table_avg <- data.frame(kFold_mat=1:nrow(parTune), parTune, JakoscT=0, JakoscW=0)
    for (i in 1:nrow(parTune)){
      result_table_avg[i, "JakoscT"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"JakoscT"])
      result_table_avg[i, "JakoscW"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)),"JakoscW"])
    }  
  }
  else {
    result_table_avg <- data.frame(mat_kfold=1:nrow(parTune), parTune, MAEt=0, MSEt=0, MAPEt=0, MAEw=0, MSEw=0, MAPEw=0)
    for (i in 1:nrow(parTune)){
      result_table_avg[i, "MAEt"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)), "MAEt"])
      result_table_avg[i, "MSEt"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)), "MSEt"])
      result_table_avg[i, "MAPEt"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)), "MAPEt"])
      result_table_avg[i, "MAEw"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)), "MAEw"])
      result_table_avg[i, "MSEw"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)), "MSEw"])
      result_table_avg[i, "MAPEw"] <- mean(result_table[seq(i, nrow(result_table), by=nrow(parTune)), "MAPEw"])
    }
  }
  return(result_table_avg) 
}

CrossValidation <- function(dane, kFold, parTune, seed, model){
  
  set.seed(seed)
  y <- dane[,1]
  idx_T <- sample(x = 1:nrow(dane), size = (1-1/kFold)*nrow(dane), replace = F)
  idx_V <- (1:nrow(dane))[-idx_T]
  
  train_valid <- vector("list", kFold)
  for(i in 1:kFold){
    tmp <- c()
    tmp[idx_T] <- 1
    tmp[idx_V] <- 2
    train_valid[[i]] <- tmp
  }
  
  kFold_mat <- matrix(1:kFold, ncol=1, nrow=nrow(parTune)*kFold)
  
  if (is.factor(y) & length(levels(y))==2){
    result_table <- data.frame(kFold_mat, parTune, AUCT=0, CzuloscT=0, SpecyficznoscT=0, JakoscT=0, AUCW=0, CzuloscW=0, SpecyficznoscW=0, JakoscW=0)
    for (i in 1:kFold){
      dane_train <- dane[which(train_valid[[i]] %in% c(1)),]
      dane_val <- dane[which(train_valid[[i]] %in% c(2)),]
      for (j in 1:nrow(parTune)){
        if (model=="KNN"){
          knn_model <- KNNtrain(dane_train[,2:ncol(dane)], dane_train[,1], k=parTune$k[j], 0, 1)
          knn_prediction_t <- KNNpred(knn_model, dane_train[,2:ncol(dane)])
          knn_prediction_v <- KNNpred(knn_model, dane_val[,2:ncol(dane)])
          ocena_t <- ModelOcena(dane_train[,1], knn_prediction_t$'1')
          ocena_v <- ModelOcena(dane_val[,1], knn_prediction_v$'1')
        }
        else if (model=='TREE'){
          tree_model <- Tree(colnames(dane)[1], colnames(dane_train[,2:ncol(dane)]), dane_train, type=parTune$type[j], depth=parTune$depth[j],
                             minobs=parTune$minobs[j], overfit='none', cf=0.05)
          tree_prediction_t <- PredTree(dane_train, tree_model, num=T)
          tree_prediction_v <- PredTree(dane_val, tree_model, num=T)
          ocena_t <- ModelOcena(dane_train[,1], tree_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], tree_prediction_v)
        }
        else if (model=="NN"){
          nn_model <- TrainNN( as.matrix(dane_train[,2:ncol(dane)]), as.matrix(as.numeric(dane_train[,1])-1), "bin", h=c(5,5), lr = parTune$lr[j], iter = parTune$iter[j], seed = seed )
          nn_prediction_t <- nn_model$y_hat
          nn_prediction_v <- PredNN(as.matrix(dane_val[,2:ncol(dane)]), nn_model, "bin")
          ocena_t <- ModelOcena(dane_train[,1], nn_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], nn_prediction_v)
        }
        result_table[j+(i-1)*nrow(parTune),"AUCT"] <- ocena_t$Miary[["AUC"]]
        result_table[j+(i-1)*nrow(parTune),"CzuloscT"] <- ocena_t$Miary[["Czulosc"]]
        result_table[j+(i-1)*nrow(parTune),"SpecyficznoscT"] <- ocena_t$Miary[["Specyficznosc"]]
        result_table[j+(i-1)*nrow(parTune),"JakoscT"] <- ocena_t$Miary[["Jakosc"]]
        result_table[j+(i-1)*nrow(parTune),"AUCW"] <- ocena_v$Miary[["AUC"]]
        result_table[j+(i-1)*nrow(parTune),"CzuloscW"] <- ocena_v$Miary[["Czulosc"]]
        result_table[j+(i-1)*nrow(parTune),"SpecyficznoscW"] <- ocena_v$Miary[["Specyficznosc"]]
        result_table[j+(i-1)*nrow(parTune),"JakoscW"] <- ocena_v$Miary[["Jakosc"]]
      }
    }
    result_table_avg <- MeanEvaluation(result_table, parTune)
  }
  else if (is.factor(y) & length(levels(y)) > 2){
    result_table <- data.frame(kFold_mat, parTune, JakoscT=0, JakoscW=0)
    for (i in 1:kFold){
      dane_train <- dane[which(train_valid[[i]] %in% c(1)),]
      dane_val <- dane[which(train_valid[[i]] %in% c(2)),]
      for (j in 1:nrow(parTune)){
        if (model=="KNN"){
          knn_model <- KNNtrain(dane_train[,2:ncol(dane)], dane_train[,1], k=parTune$k[j], 0, 1)
          knn_prediction_t <- KNNpred(knn_model, dane_train[,2:ncol(dane)])
          knn_prediction_v <- KNNpred(knn_model, dane_val[,2:ncol(dane)])
          ocena_t <- ModelOcena(dane_train[,1], knn_prediction_t$'klasa')
          ocena_v <- ModelOcena(dane_val[,1], knn_prediction_v$'klasa')
        }
        else if (model=='TREE'){
          tree_model <- Tree(colnames(dane)[1], colnames(dane_train[,2:ncol(dane)]), dane_train, type=parTune$type[j], depth=parTune$depth[j],
                             minobs=parTune$minobs[j], overfit='none', cf=0.05)
          tree_prediction_t <- PredTree(dane_train, tree_model, num=F)
          tree_prediction_v <- PredTree(dane_val, tree_model, num=F)
          ocena_t <- ModelOcena(dane_train[,1], tree_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], tree_prediction_v)
        }
        else if (model=="NN"){
          y_nn <- model.matrix( ~ dane_train[,1] - 1)
          nn_model <- TrainNN( as.matrix(dane_train[,2:ncol(dane)]), as.matrix(y_nn), "mclass", h=c(5,5), lr = parTune$lr[j], iter = parTune$iter[j], seed = seed)
          nn_prediction_t <- PredNN(as.matrix(dane_train[,2:ncol(dane)]), nn_model, "mclass")
          nn_prediction_v <- PredNN(as.matrix(dane_val[,2:ncol(dane)]), nn_model, "mclass")
          nn_pred_t <- vector("character", nrow(nn_prediction_t))
          nn_pred_v <- vector("character", nrow(nn_prediction_v))

          for (k in 1:nrow(nn_prediction_t)){
            nn_pred_t[k] <- levels(dane_train[,1])[which.max(nn_prediction_t[k,])]
          }
          
          for (k in 1:nrow(nn_prediction_v)){
            nn_pred_v[k] <- levels(dane_train[,1])[which.max(nn_prediction_v[k,])]
          }
          
          ocena_t <- ModelOcena(dane_train[,1], nn_pred_t)
          ocena_v <- ModelOcena(dane_val[,1], nn_pred_v)
        }
        result_table[j+(i-1)*nrow(parTune),"JakoscT"] <- ocena_t
        result_table[j+(i-1)*nrow(parTune),"JakoscW"] <- ocena_v
      }
    }
    result_table_avg <- MeanEvaluation(result_table, parTune)
  }
  else{
    result_table <- data.frame(kFold_mat, parTune, MAEt=0, MSEt=0, MAPEt=0, MAEw=0, MSEw=0, MAPEw=0)
    for (i in 1:kFold){
      dane_train <- dane[which(train_valid[[i]] %in% c(1)),]
      dane_val <- dane[which(train_valid[[i]] %in% c(2)),]
      for (j in 1:nrow(parTune)){
        if (model=="KNN"){
          knn_model <- KNNtrain(dane_train[,2:ncol(dane)], dane_train[,1], k=parTune$k[j], 0, 1)
          knn_prediction_t <- KNNpred(knn_model, dane_train[,2:ncol(dane)])
          knn_prediction_v <- KNNpred(knn_model, dane_val[,2:ncol(dane)])
          ocena_t <- ModelOcena(dane_train[,1], knn_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], knn_prediction_v)
        }
        else if(model=='TREE'){
          tree_model <- Tree(colnames(dane)[1], colnames(dane_train[,2:ncol(dane)]), dane_train, type=parTune$type[j], depth=parTune$depth[j],
                             minobs=parTune$minobs[j], overfit='none', cf=0.05)
          tree_prediction_t <- PredTree(dane_train, tree_model, num=T)
          tree_prediction_v <- PredTree(dane_val, tree_model, num=T)
          ocena_t <- ModelOcena(dane_train[,1], tree_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], tree_prediction_v)
        }
        else if (model=="NN"){
          nn_model <- TrainNN(as.matrix(dane_train[,2:ncol(dane)]), as.matrix(as.numeric(dane_train[,1])-1), "reg", h=c(5,5), lr = parTune$lr[j], iter = parTune$iter[j], seed = seed)
          nn_prediction_t <- nn_model$y_hat
          nn_prediction_v <- PredNN(as.matrix(dane_val[,2:ncol(dane)]), nn_model, "reg")
          ocena_t <- ModelOcena(dane_train[,1], nn_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], nn_prediction_v)
        }
        result_table[j+(i-1)*nrow(parTune),"MAEt"] <- ocena_t["MAE"]
        result_table[j+(i-1)*nrow(parTune),"MSEt"] <- ocena_t["MSE"]
        result_table[j+(i-1)*nrow(parTune),"MAPEt"] <- ocena_t["MAPE"]
        result_table[j+(i-1)*nrow(parTune),"MAEw"] <- ocena_v["MAE"]
        result_table[j+(i-1)*nrow(parTune),"MSEw"] <- ocena_v["MSE"]
        result_table[j+(i-1)*nrow(parTune),"MAPEw"] <- ocena_v["MAPE"]
      }
    }
    result_table_avg <- MeanEvaluation(result_table, parTune)
  }
  return(result_table_avg)
}

CrossValidationReadyModels <- function(dane, kFold, parTune, seed, model){
  
  set.seed(seed)
  y <- dane[,1]
  idx_T <- sample(x = 1:nrow(dane), size = (1-1/kFold)*nrow(dane), replace = F)
  idx_V <- (1:nrow(dane))[-idx_T]
  
  train_valid <- vector("list", kFold)
  for(i in 1:kFold){
    tmp <- c()
    tmp[idx_T] <- 1
    tmp[idx_V] <- 2
    train_valid[[i]] <- tmp
  }
  
  kFold_mat <- matrix(1:kFold, ncol=1, nrow=nrow(parTune)*kFold)
  
  if (is.factor(y) & length(levels(y))==2){
    result_table <- data.frame(kFold_mat, parTune, AUCT=0, CzuloscT=0, SpecyficznoscT=0, JakoscT=0, AUCW=0, CzuloscW=0, SpecyficznoscW=0, JakoscW=0)
    for (i in 1:kFold){
      dane_train <- dane[which(train_valid[[i]] %in% c(1)),]
      dane_val <- dane[which(train_valid[[i]] %in% c(2)),]
      for (j in 1:nrow(parTune)){
        if (model=="KNN"){
          knn_prediction_t <- class::knn(cl = dane_train[,1], test = dane_train[,2:ncol(dane)], train = dane_train[,2:ncol(dane)], k = parTune$k[j], prob = TRUE)
          knn_prediction_v <- class::knn(cl = dane_train[,1], test = dane_val[,2:ncol(dane)], train = dane_train[,2:ncol(dane)], k = parTune$k[j], prob = TRUE)
          ocena_t <- ModelOcena(dane_train[,1], attributes(knn_prediction_t)$prob)
          ocena_v <- ModelOcena(dane_val[,1], attributes(knn_prediction_v)$prob)
        }
        else if (model=='TREE'){
          tree_model <- rpart(formula = paste0(colnames(dane)[1], "~.", collapse=""), data = dane_train, minsplit = parTune$minobs[j], maxdepth = parTune$depth[j], parms = list(split=parTune$type[j]) )
          tree_prediction_t <- predict(tree_model, dane_train[,-1], type="prob")[,2]
          tree_prediction_v <- predict(tree_model, dane_val[,-1], type="prob")[,2]
          ocena_t <- ModelOcena(dane_train[,1], tree_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], tree_prediction_v)
          
        }
        else if (model=="NN"){
          nn_model <- nnet(formula=as.formula(paste0(colnames(dane)[1], "~.", collapse="")) ,data=dane_train ,size=5, decay=parTune$lr[j], maxit=parTune$iter)
          nn_prediction_t <- predict(nn_model, dane_train[,-1])
          nn_prediction_v <- predict(nn_model, dane_val[,-1])
          typeof(nn_prediction_t)
          ocena_t <- ModelOcena(dane_train[,1], nn_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], nn_prediction_v)
        }
        result_table[j+(i-1)*nrow(parTune),"AUCT"] <- ocena_t$Miary[["AUC"]]
        result_table[j+(i-1)*nrow(parTune),"CzuloscT"] <- ocena_t$Miary[["Czulosc"]]
        result_table[j+(i-1)*nrow(parTune),"SpecyficznoscT"] <- ocena_t$Miary[["Specyficznosc"]]
        result_table[j+(i-1)*nrow(parTune),"JakoscT"] <- ocena_t$Miary[["Jakosc"]]
        result_table[j+(i-1)*nrow(parTune),"AUCW"] <- ocena_v$Miary[["AUC"]]
        result_table[j+(i-1)*nrow(parTune),"CzuloscW"] <- ocena_v$Miary[["Czulosc"]]
        result_table[j+(i-1)*nrow(parTune),"SpecyficznoscW"] <- ocena_v$Miary[["Specyficznosc"]]
        result_table[j+(i-1)*nrow(parTune),"JakoscW"] <- ocena_v$Miary[["Jakosc"]]
      }
    }
    result_table_avg <- MeanEvaluation(result_table, parTune)
    
  }
  else if (is.factor(y) & length(levels(y)) > 2){
    result_table <- data.frame(kFold_mat, parTune, JakoscT=0, JakoscW=0)
    for (i in 1:kFold){
      dane_train <- dane[which(train_valid[[i]] %in% c(1)),]
      dane_val <- dane[which(train_valid[[i]] %in% c(2)),]
      for (j in 1:nrow(parTune)){
        if (model=="KNN"){
          knn_prediction_t <- class::knn(train=dane_train[,2:ncol(dane)], test=dane_train[,2:ncol(dane)], cl=dane_train[,1], k=parTune$k[j])
          knn_prediction_v <- class::knn(train=dane_train[,2:ncol(dane)], test=dane_val[,2:ncol(dane)], cl=dane_train[,1], k=parTune$k[j])
          ocena_t <- ModelOcena(dane_train[,1], knn_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], knn_prediction_v)
        }
        else if (model=='TREE'){
          f <- paste0(colnames(dane)[1], "~.", collapse="")
          tree_model <- rpart( formula = f, data = dane_train, minsplit = parTune$minobs[j], maxdepth = parTune$depth[j], parms = list(split=parTune$type[j]) )
          tree_prediction_t <- predict(tree_model, dane_train[,2:ncol(dane)], type="class")
          tree_prediction_v <- predict(tree_model, dane_val[,2:ncol(dane)], type="class")
          ocena_t <- ModelOcena(dane_train[,1], tree_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], tree_prediction_v)
        }
        else if (model=="NN"){
          nn_model <- nnet(formula=as.formula(paste0(colnames(dane)[1], "~.", collapse="")), data=dane_train ,size=5, decay=parTune$lr[j], maxit=parTune$iter[j])
          nn_prediction_t <- predict(nn_model, dane_train[,-1], type="class")
          nn_prediction_v <- predict(nn_model, dane_val[,-1], type="class")
          ocena_t <- ModelOcena(dane_train[,1], nn_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], nn_prediction_v)
        }
        result_table[j+(i-1)*nrow(parTune),"JakoscT"] <- ocena_t
        result_table[j+(i-1)*nrow(parTune),"JakoscW"] <- ocena_v
      }
    }
    result_table_avg <- MeanEvaluation(result_table, parTune)
  }
  else{
    result_table <- data.frame(kFold_mat, parTune, MAEt=0, MSEt=0, MAPEt=0, MAEw=0, MSEw=0, MAPEw=0)
    for (i in 1:kFold){
      dane_train <- dane[which(train_valid[[i]] %in% c(1)),]
      dane_val <- dane[which(train_valid[[i]] %in% c(2)),]
      for (j in 1:nrow(parTune)){
        if (model=="KNN"){
          knn_prediction_t <- knn.reg(train=dane_train[,2:ncol(dane)], y=dane_train[,1], test=dane_train[,2:ncol(dane)], k = parTune$k[j])
          knn_prediction_v <- knn.reg(train=dane_train[,2:ncol(dane)], y=dane_train[,1], test=dane_val[,2:ncol(dane)], k = parTune$k[j])
          ocena_t <- ModelOcena(dane_train[,1], knn_prediction_t$pred)
          ocena_v <- ModelOcena(dane_val[,1], knn_prediction_v$pred)
        }
        else if(model=='TREE'){
          tree_model <- rpart(formula = paste0(colnames(dane)[1], "~.", collapse=""), data = dane_train, minsplit = parTune$minobs[j], maxdepth = parTune$depth[j])
          tree_prediction_t <- predict(tree_model, dane_train[,2:ncol(dane)])
          tree_prediction_v <- predict(tree_model, dane_val[,2:ncol(dane)])
          ocena_t <- ModelOcena(dane_train[,1], tree_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], tree_prediction_v)
        }
        else if (model=="NN"){
          nn_model <- nnet(formula=as.formula(paste0(colnames(dane)[1], "~.", collapse="")), data=dane_train,size=5, decay=parTune$lr[j], maxit=parTune$iter[j], linout=TRUE)
          nn_prediction_t <- predict(nn_model, dane_train[,2:ncol(dane)])
          nn_prediction_v <- predict(nn_model, dane_val[,2:ncol(dane)])
          ocena_t <- ModelOcena(dane_train[,1], nn_prediction_t)
          ocena_v <- ModelOcena(dane_val[,1], nn_prediction_v)
        }
        result_table[j+(i-1)*nrow(parTune),"MAEt"] <- ocena_t["MAE"]
        result_table[j+(i-1)*nrow(parTune),"MSEt"] <- ocena_t["MSE"]
        result_table[j+(i-1)*nrow(parTune),"MAPEt"] <- ocena_t["MAPE"]
        result_table[j+(i-1)*nrow(parTune),"MAEw"] <- ocena_v["MAE"]
        result_table[j+(i-1)*nrow(parTune),"MSEw"] <- ocena_v["MSE"]
        result_table[j+(i-1)*nrow(parTune),"MAPEw"] <- ocena_v["MAPE"]
      }
    }
    result_table_avg <- MeanEvaluation(result_table, parTune)
  }
  return(result_table_avg)
}

SelectBestModelCV <- function(df, model_type){
  
  df = cbind(model=model_type, df)
  if ('MAPEw' %in% colnames(df)){
    cat("\n Oceny modeli po kroswalidacji: \n")
    print(df)
    cat("\n Najlepszy model: \n")
    best <- df[which.min(df$MAPEw),]
    print(best)
    return(best[,c('model', 'MAPEw')])
  }
  else if ('JakoscW' %in% colnames(df)){
    cat("\n Oceny modeli po kroswalidacji: \n")
    print(df)
    cat("\n Najlepszy model: \n")
    best <- df[which.max(df$JakoscW),]
    print(best)
    return(best[,c('model', 'JakoscW')])
  }
  else{
    stop()
  }
}

SelectBestModel <- function(df){
  
  if ('MAPEw' %in% colnames(df)){
    cat("\n Oceny najlepszych algorytmów: \n")
    print(df)
    cat("\n Najlepszy najlepszy algorytm: \n")
    best <- df[which.min(df$MAPEw),]
    print(best)
  }
  else if ('JakoscW' %in% colnames(df)){
    cat("\n Oceny najlepszych algorytmów: \n")
    print(df)
    cat("\n Najlepszy najlepszy algorytm: \n")
    best <- df[which.max(df$JakoscW),]
    print(best)
  }
  else{
    stop()
  }
}