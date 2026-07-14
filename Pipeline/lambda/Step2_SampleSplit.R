library(dplyr)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript your_script.R <seed>")
}
my_seed <- as.numeric(args[1])
cat("Using seed:", my_seed, "\n")
Neg <- read.csv("./1.Data/Neg/6.Neg_igblast_table.xls",sep = "\t",header = F)
Neg$type <- "Neg"
Pos <- read.csv("./1.Data/Pos/6.Pos_igblast_table.xls",sep = "\t",header = F)
Pos$type <- "Pos"
# 按第二列（V2）去重，保留每组的第一行
unique_Neg <- Neg %>%
  distinct(V2, .keep_all = TRUE)
unique_Pos <- Pos %>%
  distinct(V2, .keep_all = TRUE)
all_data <- rbind(unique_Neg,unique_Pos)
fina_table <- data.frame(ID = all_data$V2,Gene = all_data$V3,Type = all_data$type,Ifkappa = 0)
fina_table[grep("IGK",fina_table$Gene),4] <- 1
table(fina_table$Type,fina_table$Ifkappa)
Neg_fasta <- read.csv("./1.Data/Neg/4.cdhit-Neg_seq_fv.tsv",sep ="\t")
Pos_fasta <- read.csv("./1.Data/Pos/4.cdhit-Pos_seq_fv.tsv",sep ="\t")
All_fasta<- rbind(Pos_fasta,Neg_fasta)
fina_table$fasta <- All_fasta[match(fina_table$ID,All_fasta$ID),2]
Neg_kappa <- fina_table[fina_table$Type =="Neg"&fina_table$Ifkappa==1,]
Neg_lambda <- fina_table[fina_table$Type =="Neg"&fina_table$Ifkappa==0,]
Pos_kappa <- fina_table[fina_table$Type =="Pos"&fina_table$Ifkappa==1,]
Pos_lambda <- fina_table[fina_table$Type =="Pos"&fina_table$Ifkappa==0,]
write.table(fina_table,"./1.Data/CleanData_info.xls",sep = "\t",col.names = T,row.names = F,quote = F)
## 数据划分-lambda
# 使用函数写出文件
write_fasta <- function(df, file) {
  con <- file(file, "w")
  for (i in 1:nrow(df)) {
    writeLines(paste0(">", df[i, 1]), con)        # 写入ID行（以 > 开头）
    writeLines(df[i, 5], con)                     # 写入序列行
  }
  close(con)
}

set.seed(my_seed)
sample_pos <- sample(1:nrow(Pos_lambda),122)
test_Pos_lambda <- Pos_lambda[sample_pos,]
sample_neg <- sample(1:nrow(Neg_lambda),122)
test_Neg_lambda <- Neg_lambda[sample_neg,]
test_Pos_lambda$set <- "test"
test_Neg_lambda$set <- "test"
# 并删除取出的样本
Pos_lambda <- Pos_lambda[-sample_pos,]
Neg_lambda <- Neg_lambda[-sample_neg,]
write_fasta(Pos_lambda, "./1.Data/Lambda_pos_train_490.fasta")
write_fasta(test_Pos_lambda, "./1.Data/Lambda_pos_test_122.fasta")
write_fasta(test_Neg_lambda, "./1.Data/Lambda_neg_test_122.fasta")
train_name_l <- c()
for (i in 1:1){
    set.seed(my_seed)
    sample_neg <- sample(1:nrow(Neg_lambda),490)
    train_Neg_lambda <- Neg_lambda[sample_neg,]
    id <- cbind(train_Neg_lambda$ID,paste0("trainset-",i))
    train_name_l <- rbind(train_name_l,id)
    filename <- paste0("./1.Data/Lambda_neg_trainset490-",i,".fasta")
    write_fasta(train_Neg_lambda, filename)
    train_all <- rbind(Pos_lambda,train_Neg_lambda)
    filename2  <- paste0("./2.Training_Data/Lambda/Lambda_trainset-",i,"_label.txt")
    write.table(data.frame(Name = train_all$ID,label = ifelse(train_all$Type=="Pos",1,0)),
                file = filename2,sep ="\t",col.names = T,row.names = F,quote = F)
    filename3 <- paste0("./2.Training_Data/Lambda/Lambda_trainset-",i,".fasta")
    write_fasta(train_all, filename3)
    Neg_lambda <- Neg_lambda[-sample_neg,]
}
## 1.3  组合正负样本
system("mkdir 2.Training_Data/Lambda", intern = TRUE)
system("cat ./1.Data/Lambda_pos_test_122.fasta ./1.Data/Lambda_neg_test_122.fasta > 2.Training_Data/Lambda/Lambda_test_244.fasta", intern = TRUE)
label_lambda_test <- rbind(test_Pos_lambda,test_Neg_lambda)
write.table(data.frame(Name = label_lambda_test$ID,label = ifelse(label_lambda_test$Type=="Pos",1,0)),
            file = "2.Training_Data/Lambda/Lambda_test_244_label.txt",sep ="\t",col.names = T,row.names = F,quote = F)
Neg_lambda <- fina_table[fina_table$Type =="Neg"&fina_table$Ifkappa==0,]
Pos_lambda <- fina_table[fina_table$Type =="Pos"&fina_table$Ifkappa==0,]
# 取百分之20
set.seed(my_seed)
sample_pos <- sample(1:nrow(Pos_lambda),122)
test_Pos_lambda <- Pos_lambda[sample_pos,]
sample_neg <- sample(1:nrow(Neg_lambda),122)
test_Neg_lambda <- Neg_lambda[sample_neg,]
test_Pos_lambda$set <- "test"
test_Neg_lambda$set <- "test"
# 并删除取出的样本
Pos_lambda <- Pos_lambda[-sample_pos,]
Neg_lambda <- Neg_lambda[-sample_neg,]
dim(Pos_lambda)
sample_pos
lambda_all_tr<- rbind(Pos_lambda,Neg_lambda[match(train_name_l[,1],Neg_lambda$ID),])
write.table(data.frame(Name = lambda_all_tr$ID,label = ifelse(lambda_all_tr$Type=="Pos",1,0)),
            file = "2.Training_Data/Lambda/Lambda_all_tr1470_label.txt",sep ="\t",col.names = T,row.names = F,quote = F)
write_fasta(lambda_all_tr, "2.Training_Data/Lambda/Lambda_all_tr1470.fasta")
