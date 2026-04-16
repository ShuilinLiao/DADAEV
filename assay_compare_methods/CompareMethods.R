# https://github.com/HuangLab-Fudan/EV-origin
EVOriginMet <- function(data2.m, ref2.m){
  library(e1071)
  nu.v = c(0.25, 0.5, 0.75)
  
  est.ab.lm <- list()
  est.lm <- list()
  
  nui <- 1
  for (nu in nu.v) {
    # nu = 0.25
    est.m <- matrix(NA, nrow = ncol(data2.m), ncol = ncol(ref2.m))
    colnames(est.m) <- colnames(ref2.m)
    rownames(est.m) <- colnames(data2.m)
    est.ab.m <- matrix(NA, nrow = ncol(data2.m), ncol = ncol(ref2.m))
    colnames(est.ab.m) <- colnames(ref2.m)
    rownames(est.ab.m) <- colnames(data2.m)
    
    for (s in seq_len(ncol(data2.m))) {
      svm.o <- svm(x = ref2.m, y = data2.m[, s], scale = TRUE, type = "nu-regression", kernel = "linear", nu = nu)
      coef.v <- t(svm.o$coefs) %*% svm.o$SV
      coef.v[which(coef.v < 0)] <- 1*10^-10
      est.ab.m[s,] <- coef.v
      total <- sum(coef.v)
      coef.v <- coef.v/total
      est.m[s, ] <- coef.v
    }
    est.lm[[nui]] <- est.m
    est.ab.lm[[nui]] <- est.ab.m
    nui <- nui + 1
  }
  
  ## select best nu using RMSE
  rmse.m <- matrix(NA, nrow = ncol(data2.m), ncol = length(nu.v))
  for (nui in seq_along(nu.v)) {
    reconst.m <- (ref2.m %>% as.matrix()) %*% t(est.lm[[nui]])
    s <- seq_len(ncol(data2.m))
    rmse.m[s, nui] <- sqrt(colMeans((data2.m[, s] - reconst.m[, s])^2))
  }
  colnames(rmse.m) <- nu.v
  nu.idx <- apply(rmse.m, 1, which.min)
  
  estF.m <- est.m
  for (s in seq_len(nrow(estF.m))) {
    estF.m[s, ] <- est.lm[[nu.idx[s]]][s, ]
  }
  
  estF.ab.m <- est.ab.m
  for (s in seq_len(nrow(estF.ab.m))) {
    estF.ab.m[s, ] <- est.ab.lm[[nu.idx[s]]][s, ]
  }
  
  return(list(estF.m = estF.m %>% t(), estF.ab.m = estF.ab.m %>% t()))
}

# ?
CIBERSORTxMet <- function(data2.m, ref2.m){


  

}

# https://github.com/gjhunt/dtangle/tree/master
# https://gjhunt.github.io/dtangle/vign/basic-deconvolution.html
DtangleMet <- function(data2.m, ref2.m){
  library('dtangle')
  
  # load("../data/compare_method_data/dtangle/shen_orr_ex.rda")
  # data = shen_orr_ex$data$log
  # mixture_proportions = shen_orr_ex$annotation$mixture
  # ref_reduced = t(sapply(pure_samples,function(x)colMeans(reference_samples[x,,drop=FALSE])))
  # dt_out = dtangle(Y=mixture_samples, reference=ref_reduced)
  # matplot(mixture_mixture_proportions,dt_out$estimates, xlim = c(0,1),ylim=c(0,1),xlab="Truth",ylab="Estimates")
   
  mixture_samples = data2.m %>% t() %>% as.matrix()
  ref_reduced = ref2.m %>% t() %>% as.matrix()
  dt_out = dtangle(Y=mixture_samples, reference=ref_reduced)
  estF.m <- dt_out$estimates %>% t() %>% data.frame()
  return(estF.m)
}

# https://github.com/cozygene/bisque/tree/master
BisqueMet <- function(data2.m, ref2.m){
  library(BisqueRNA)
  library(Biobase)
  
  ##============ simulate (Bisque Example) =================
  # set.seed(42)
  # cell.types <- c("Neurons", "Astrocytes", "Oligodendrocytes", "Microglia", "Endothelial Cells")
  # avg.props <- c(.5, .2, .2, .07, .03)
  # sim.data <- SimulateData(n.ind=10, n.genes=100, n.cells=500, cell.types=cell.types, avg.props=avg.props)
  # sc.eset <- sim.data$sc.eset[,sim.data$sc.eset$SubjectName %in% as.character(6:10)]
  # bulk.eset <- sim.data$bulk.eset
  # true.props <- sim.data$props
  # markers <- sim.data$markers
  # 
  # # reference_based
  # # 该方法要求使用同一样本的两种不同测序数据
  # res <- BisqueRNA::ReferenceBasedDecomposition(bulk.eset, sc.eset, markers=NULL, use.overlap=TRUE)
  # 
  # # Marker-based
  # marker.data.frame <- data.frame(gene=paste("Gene", 1:6),
  #                                 cluster=c("Neurons", "Neurons", "Astrocytes", "Oligodendrocytes", "Microglia", "Endothelial Cells"),
  #                                 avg_logFC=c(0.82, 0.59, 0.68, 0.66, 0.71, 0.62))
  # res <- BisqueRNA::MarkerBasedDecomposition(bulk.eset, markers, weighted=F)
  
  ##============ our data (input data) =================
  bulk.eset <- Biobase::ExpressionSet(assayData = data2.m)
  
  sc.pheno <- data.frame(check.names=F, check.rows=F,
                         stringsAsFactors=F,
                         row.names=colnames(ref2.m),
                         SubjectName=colnames(ref2.m),
                         cellType=colnames(ref2.m))
  
  sc.meta <- data.frame(labelDescription=c("SubjectName", "cellType"),
                        row.names=c("SubjectName", "cellType"))
  sc.pdata <- new("AnnotatedDataFrame", data=sc.pheno, varMetadata=sc.meta)
  sc.eset <- Biobase::ExpressionSet(assayData = ref2.m %>% as.matrix(), phenoData=sc.pdata)
  
  res <- BisqueRNA::ReferenceBasedDecomposition(bulk.eset, sc.eset, markers=NULL, use.overlap=F)
  estF.m <- res$bulk.props
  return(estF.m)
}

# https://bioconductor.org/packages/release/bioc/html/DeconRNASeq.html
DeconRNASeqMet <- function(data2.m, ref2.m){
  library(DeconRNASeq)
  datasets <- data2.m %>% data.frame()
  signatures <- ref2.m %>% data.frame()
  res <- DeconRNASeq(datasets, signatures, proportions = NULL, checksig=FALSE, 
              known.prop = F, use.scale = TRUE, fig = TRUE)
  estF.m = res[["out.all"]] %>% t() %>% data.frame()
  colnames(estF.m) <- colnames(datasets)
  
  return(estF.m)
}

# https://github.com/dtsoucas/DWLS/
DWLSMet <- function(data2.m, ref2.m){
  source("/disk1/user/liaoshuilin/project/35.TAPE_EXO/assay_compare_methods/Deconvolution_functions.R")
  
  # data2.m = stim_gtex_x
  
  sig = as.matrix(ref2.m)
  solDWLS <- matrix(NA, nrow = ncol(data2.m), ncol=ncol(ref2.m)) %>% data.frame()
  solSVR <- matrix(NA, nrow = ncol(data2.m), ncol=ncol(ref2.m)) %>% data.frame()
  solOLS <- matrix(NA, nrow = ncol(data2.m), ncol=ncol(ref2.m)) %>% data.frame()
  for(i in 1:nrow(solDWLS)){
    # print(i)
    dataBulk = data2.m[,i]
    names(dataBulk) = rownames(data2.m)
    tr <- trimData(sig, dataBulk)
    tr_sig = tr$sig
    tr_bulk = tr$bulk
    solDWLS_res = solveDampenedWLS(tr_sig, tr_bulk)
    solSVR_res = solveSVR(tr_sig, tr_bulk)
    solOLS_res = solveOLS(tr_sig, tr_bulk)
    solDWLS[i,] = solDWLS_res
    solSVR[i,] = solSVR_res
    solOLS[i,] = solOLS_res
  }
  
  # solDWLS <- apply(data2.m, 2, function(x) solveDampenedWLS(sig, x)) %>% data.frame() # DWLS
  # solSVR <- apply(data2.m, 2, function(x) solveSVR(sig, x)) %>% data.frame() # DWLS SVR
  # solOLS <- apply(data2.m, 2, function(x) solveOLS(sig, x)) %>% data.frame() # DWLS OLS
  
  return(list(solDWLS = solDWLS, solSVR = solSVR, solOLS = solOLS))
}

# https://github.com/xuranw/MuSiC
# https://xuranw.github.io/MuSiC/articles/MuSiC.html
MusicMet <- function(data2.m, ref2.m){
  
}

# Evaluation criteria
CalculateEvaMet_x <- function(data2.m, ref2.m = NULL, estF.m = NULL, reconst.m = NULL, x_state = "unknown"){
  
  # data2.m = stim_hpa_x
  # estF.m = res_stim_hpa_EVOrigin[["estF.m"]]
  # x_state = "unknown"
  
  if(x_state == "unknown"){
    ## 针对未知的重建的x
    reconst.m <- as.matrix(ref2.m) %*% as.matrix(estF.m)
  }else{
    ## 针对已知的重建的x或frac
    reconst.m <- reconst.m
  }
  
  ## caculating RMSE
  rmse.m <- sqrt(colMeans((data2.m - reconst.m)^2))
  RMSE = rmse.m
  
  ## caculating PCC
  pearson.corr.value <- c()
  for (i in 1:ncol(data2.m)) {
    cor.index <- cor.test(data2.m[, i], reconst.m[, i])
    cor.p <- as.numeric(cor.index$estimate)
    pearson.corr.value <- c(pearson.corr.value, cor.p)
  }
  PCC = pearson.corr.value
  
  ## calculating MAE (Mean Absolute Error)
  mae.m <- colMeans(abs(data2.m - reconst.m))
  MAE = mae.m
  
  ## calculating Lin's Concordance Correlation Coefficient (CCC)
  ccc.m <- c()
  for (i in 1:ncol(data2.m)) {
    # Calculate means and variances
    mean_data <- mean(data2.m[, i])
    mean_reconst <- mean(reconst.m[, i])
    var_data <- var(data2.m[, i])
    var_reconst <- var(reconst.m[, i])
    covar <- cov(data2.m[, i], reconst.m[, i])
    
    # Calculate CCC
    ccc <- (2 * covar) / (var_data + var_reconst + (mean_data - mean_reconst)^2)
    ccc.m <- c(ccc.m, ccc)
  }
  CCC = ccc.m
  ## Add PCC and RMSE to the final estimation
  val_mrx <- cbind.data.frame(RMSE, PCC, MAE, CCC)
  return(val_mrx)
}

CalculateEvaMet_f <- function(real.m, estF.m){
  # estF.m = res_stim_hpa_EVOrigin[["estF.m"]]
  # real.m = stim_hpa_y
  
  data2.m = real.m
  reconst.m = estF.m
  
  ## caculating RMSE
  rmse.m <- sqrt(colMeans((data2.m - reconst.m)^2))
  
  ## caculating PCC
  pearson.corr.value <- c()
  for (i in 1:ncol(data2.m)) {
    cor.index <- cor.test(data2.m[, i], reconst.m[, i])
    cor.p <- as.numeric(cor.index$estimate)
    pearson.corr.value <- c(pearson.corr.value, cor.p)
  }
  
  ## calculating MAE (Mean Absolute Error)
  mae.m <- colMeans(abs(data2.m - reconst.m))
  
  ## calculating Lin's Concordance Correlation Coefficient (CCC)
  ccc.m <- c()
  for (i in 1:ncol(data2.m)) {
    # Calculate means and variances
    mean_data <- mean(data2.m[, i])
    mean_reconst <- mean(reconst.m[, i])
    var_data <- var(data2.m[, i])
    var_reconst <- var(reconst.m[, i])
    covar <- cov(data2.m[, i], reconst.m[, i])
    
    # Calculate CCC
    ccc <- (2 * covar) / (var_data + var_reconst + (mean_data - mean_reconst)^2)
    ccc.m <- c(ccc.m, ccc)
  }
  
  ## Add PCC and RMSE to the final estimation
  val_mrx <- cbind.data.frame(rmse.m, pearson.corr.value, mae.m, ccc.m)
  return(val_mrx)
}
