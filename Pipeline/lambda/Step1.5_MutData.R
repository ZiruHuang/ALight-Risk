library(stringr)
library(dplyr)

IgblastToMutationTable <- function(filename){

  lines <- readLines(filename)

  query_idx <- grep("^Query=", lines)
  query_idx <- c(query_idx, length(lines) + 1)

  parse_one_query <- function(block_lines) {

    query_line <- block_lines[grep("^Query=", block_lines)[1]]
    protein_id <- sub("Query=\\s*", "", query_line)

    aln_start <- grep("^Alignments", block_lines)
    if (length(aln_start) == 0) return(NULL)

    aln_lines <- block_lines[(aln_start + 1):length(block_lines)]

    query_seq_all <- c()
    germ_seq_all  <- c()
    query_pos_all <- c()

    i <- 1
    while (i <= length(aln_lines)) {

      line <- aln_lines[i]

      if (grepl("^\\s*Query", line)) {

        parts <- strsplit(line, "\\s+")[[1]]
        parts <- parts[parts != ""]

        query_start <- as.numeric(parts[2])
        query_seq   <- parts[3]

        # 找germline
        j <- i + 1
        germ_seq <- NULL

        while (j <= length(aln_lines)) {
          if (grepl("^V\\s+", aln_lines[j])) {
            parts_g <- strsplit(aln_lines[j], "\\s+")[[1]]
            parts_g <- parts_g[parts_g != ""]
            germ_seq <- parts_g[length(parts_g) - 1]
            break
          }
          j <- j + 1
        }

        if (is.null(germ_seq)) {
          i <- i + 1
          next
        }

        q_chars <- strsplit(query_seq, "")[[1]]
        g_chars <- strsplit(germ_seq, "")[[1]]

        pos_counter <- query_start

        for (k in seq_along(q_chars)) {

          q_aa <- q_chars[k]
          g_aa <- g_chars[k]

          #  如果是gap（query中），跳过，不计入position
          if (q_aa == "-") {
            next
          }

          # 记录
          query_seq_all <- c(query_seq_all, q_aa)
          germ_seq_all  <- c(germ_seq_all, g_aa)
          query_pos_all <- c(query_pos_all, pos_counter)

          pos_counter <- pos_counter + 1
        }
      }

      i <- i + 1
    }

    # 转换
    q <- query_seq_all
    g <- germ_seq_all

    if (length(q) != length(g) || length(q) != length(query_pos_all)) {
      warning(paste("长度不一致:", protein_id))
      return(NULL)
    }

    # 处理 germline
    germ_seq <- ifelse(g == ".", q, g)

    #  删除 germline gap（代表 insertion）
    valid_idx <- germ_seq != "-"
    q <- q[valid_idx]
    germ_seq <- germ_seq[valid_idx]
    query_pos_all <- query_pos_all[valid_idx]

    df <- data.frame(
      protein_id = protein_id,
      position   = query_pos_all,
      germ_aa    = germ_seq,
      obs_aa     = q,
      stringsAsFactors = FALSE
    )

    df <- df %>%
      filter(germ_aa != obs_aa) %>%
      mutate(mutation = paste0(germ_aa, ">", obs_aa))

    return(df)
  }

  mut_list <- list()

  for (i in 1:(length(query_idx)-1)) {
    block <- lines[query_idx[i]:(query_idx[i+1]-1)]
    res <- parse_one_query(block)
    if (!is.null(res)) {
      mut_list[[length(mut_list)+1]] <- res
    }
  }

  mut_df <- bind_rows(mut_list)
  return(mut_df)
}
al_mut <- IgblastToMutationTable("./1.Data/Pos_seq_igblast.out")
write.table(al_mut,file = "./1.Data/Pos_mutation_table.csv",sep =",",row.names=F,col.names = T,quote = F)
nonal_mut <- IgblastToMutationTable("./1.Data/Neg_seq_igblast.out")
write.table(nonal_mut,file = "./1.Data/Neg_mutation_table.csv",sep =",",row.names=F,col.names = T,quote = F)