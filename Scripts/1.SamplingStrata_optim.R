# ================================================================================
#  OPTIMAL STRATIFICATION - SamplingStrata (classical baseline)
# ================================================================================
#  Builds the classical (non-quantum) benchmark for the stratification study.
#  Uses the SamplingStrata package on the "swissmunicipalities" dataset (region
#  REG == 1) to: bin the stratification variables (POPTOT, HApoly), build the
#  sampling frame and atomic strata, compute the Bethel allocation for a set of
#  precision constraints (CV), find an initial K-means solution, and run the
#  genetic-algorithm optimization (optimStrata) to obtain the final number of
#  strata and sample size. The resulting atomic strata are exported to
#  "swiss_municipalities.csv", used as the common input frame by all the
#  quantum stratification scripts in this folder.
# ================================================================================

library(SamplingStrata)

data("swissmunicipalities")
swissmun <- swissmunicipalities[swissmunicipalities$REG == 1,
                                c("REG","COM","Nom","HApoly",
                                  "Surfacesbois","Surfacescult",
                                  "Airbat","POPTOT")]

corrplot::corrplot(cor(swissmun[,c("HApoly","Surfacesbois","Surfacescult","Airbat","POPTOT")]))

swissmun$HApoly.cat <- var.bin(swissmun$HApoly,5)
table(swissmun$HApoly.cat)
swissmun$POPTOT.cat <- var.bin(swissmun$POPTOT,5)
table(swissmun$POPTOT.cat)

swissmun$DOM <- 1
frame <- buildFrameDF(df = swissmun,
                       id = "COM",
                       X = c("POPTOT.cat","HApoly.cat"),
                       Y = c("Airbat","Surfacesbois"),
                       domainvalue = "DOM")

head(frame)


# Define precision constraints ------------------------------------------------------------
ndom <- length(unique(frame$domainvalue))
cv <- as.data.frame(list(DOM = rep("DOM1",ndom),
                         CV1 = rep(0.05,ndom),      # precision (cv=10%) for 'Surfacesbois'
                         CV2 = rep(0.05,ndom),      # precision (cv=10%) for 'Airind'
                         domainvalue= c(1:ndom)))   # same precision constraints for all domains
cv
strata <- buildStrataDF(frame)
head(strata)
nrow(strata)
# [1] 20
alloc <-bethel(strata,cv)
sum(alloc)
# [1] 105
# Find an initial solution and a suitable number of final strata in each domain -----------
set.seed(123)
solutionKmean <- KmeansSolution(strata = strata,    # atomic strata
                                errors = cv,        # precision constraints
                                maxclusters = 4)   # max number of strata to be evaluated 
# -------------------
#   Kmeans solution 
# -------------------
#   *** Domain:  1  ***
#   Number of strata:  4
# Sample size     :  312
# -------------------
#   Total size:  312
# -------------------
# number of strata to be obtained in each domain in final solution:                             
nstrat <- tapply(solutionKmean$suggestions, solutionKmean$domainvalue,
                 FUN=function(x) length(unique(x)))

# Optimization step ------------------------------------------------------------------------
set.seed(1234)
solution <- optimStrata(method = "atomic",          # method
                        framesamp = frame,            # sampling frame
                        errors = cv,                  # precision constraints
                        nStrata = 4,             # strata to be obtained in the final stratification
                        # suggestions = solutionKmean,  # initial solution
                        iter = 500,                    # number of iterations
                        pops = 10)                    # number of stratifications evaluated at each iteration
# *** Sample size :  129
# *** Number of strata :  4
# ---------------------------

write.table(strata,"swiss_municipalities.csv",sep=",",quote=F,row.names = F)
