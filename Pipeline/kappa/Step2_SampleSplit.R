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

Pos_kappa <- fina_table[fina_table$Type =="Pos"&fina_table$Ifkappa==1,]

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



### kappa
# 取百分之20
set.seed(my_seed)
sample_pos <- sample(1:nrow(Pos_kappa),44)
test_Pos_kappa <- Pos_kappa[sample_pos,]
sample_neg <- sample(1:nrow(Neg_kappa),44)
test_Neg_kappa <- Neg_kappa[sample_neg,]
test_Pos_kappa$set <- "test"
test_Neg_kappa$set <- "test"
# 并删除取出的样本
Pos_kappa <- Pos_kappa[-sample_pos,]
Neg_kappa <- Neg_kappa[-sample_neg,]
dim(Pos_kappa)
sample_pos

# 使用函数写出文件
write_fasta <- function(df, file) {
  con <- file(file, "w")
  for (i in 1:nrow(df)) {
    writeLines(paste0(">", df[i, 1]), con)        # 写入ID行（以 > 开头）
    writeLines(df[i, 5], con)                     # 写入序列行
  }
  close(con)
}

write_fasta(Pos_kappa, "./1.Data/Kappa_pos_train_178.fasta")
write_fasta(test_Pos_kappa, "./1.Data/Kappa_pos_test_44.fasta")
write_fasta(test_Neg_kappa, "./1.Data/Kappa_neg_test_44.fasta")


train_name_k <- c()
for (i in 1:1){
    set.seed(my_seed)
    sample_neg <- sample(1:nrow(Neg_kappa),178)
    train_Neg_kappa <- Neg_kappa[sample_neg,]
    id <- cbind(train_Neg_kappa$ID,paste0("trainset-",i))
    train_name_k <- rbind(train_name_k,id)
    ## 
    filename <- paste0("./1.Data/Kappa_neg_trainset178-",i,".fasta")
    train_all <- rbind(Pos_kappa,train_Neg_kappa)
    write_fasta(train_Neg_kappa, filename)
    filename2  <- paste0("./2.Training_Data/Kappa/Kappa_trainset-",i,"_label.txt")
    write.table(data.frame(Name = train_all$ID,label = ifelse(train_all$Type=="Pos",1,0)),
            file = filename2,sep ="\t",col.names = T,row.names = F,quote = F)
    filename3 <- paste0("./2.Training_Data/Kappa/Kappa_trainset-",i,".fasta")
    write_fasta(train_all, filename3)
    Neg_kappa <- Neg_kappa[-sample_neg,]
}

system("mkdir 2.Training_Data/Kappa ", intern = TRUE)
system("cat ./1.Data/Kappa_pos_test_44.fasta ./1.Data/Kappa_neg_test_44.fasta > 2.Training_Data/Kappa/Kappa_test_88.fasta", intern = TRUE)
label_kappa_test <- rbind(test_Pos_kappa,test_Neg_kappa)
write.table(data.frame(Name = label_kappa_test$ID,label = ifelse(label_kappa_test$Type=="Pos",1,0)),
            file = "2.Training_Data/Kappa/Kappa_test_88_label.txt",sep ="\t",col.names = T,row.names = F,quote = F)

