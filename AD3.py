#Practical 3
#Text Analytics
#LDA
#code:

library(tm)
library(topicmodels)

setwd("C:/Users/Admin/Downloads/british-fiction-corpus/british-fiction-corpus")

filenames <-list.files(path=".", pattern ="*.txt")
filenames

filetext <-lapply(filenames, readLines)
mycorpus <-Corpus(VectorSource(filetext))

mycorpus <- tm_map(mycorpus, content_transformer(tolower))
mycorpus <- tm_map(mycorpus, removeNumbers)
mycorpus <- tm_map(mycorpus, removePunctuation)

mystopwords <- c("of", "a", "and", "the", "in", "to", "for", "that", "is", "on", "are", 
                 "with", "as", "by", "be", "an", "which", "it", "from", "or", "can", 
                 "have", "these", "has", "such")

mycorpus <- tm_map(mycorpus, removeWords, mystopwords)

dtm <- DocumentTermMatrix(mycorpus)

inspect(dtm)

k<-3
lda_output <- LDA(dtm, k, method = "VEM")
print(k)

summary(lda_output)
topics(lda_output)
terms(lda_output, 10)
posterior(lda_output)$topics

topic_terms<-terms(lda_output, 10)
print(topic_terms)

#2] Sentiment Analysis:
# code:
weatherdata <- read.csv("C:/Users/Admin/Downloads/weather.csv")

# Load the required libraries
install.packages("wordcloud")
install.packages("RColorBrewer")
install.packages("syuzhet")
install.packages("lubridate")
install.packages("scales")
install.packages("reshape2")
install.packages("dplyr")

library(tm)
library(wordcloud)
library(RColorBrewer)  # For color palettes
library(syuzhet)
library(lubridate)
library(ggplot2)
library(scales)
library(reshape2)
library(dplyr)

corpus <- iconv(weatherdata$tweet_text, to = "utf-8")
corpus <- Corpus(VectorSource(corpus))

# Text preprocessing
corpus <- tm_map(corpus, content_transformer(tolower))        # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)                    # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)                        # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords('english'))    # Remove stopwords
corpus <- tm_map(corpus, stripWhitespace)                      # Remove extra whitespace

inspect(corpus[1:5])

tdm <- TermDocumentMatrix(corpus)

tdm_matrix <- as.matrix(tdm)
tdm_matrix[1:10, 1:2]

w <- rowSums(tdm_matrix)
w <- sort(w, decreasing = TRUE)

set.seed(222)
wordcloud(words = names(w), freq = w, max.words = 150, random.order = FALSE, min.freq = 5, colors = brewer.pal(8, 'Dark2'))

wordcloud(words = names(w), freq = w, max.words = 150, random.order = FALSE, min.freq = 5, colors = brewer.pal(8, 'Dark2'), rot.per = 0.3)

weatherdata <- read.csv("C:/Users/Admin/Downloads/weather.csv")
tweets <- iconv(weatherdata$tweet_text, to = "utf-8")

s <- get_nrc_sentiment(tweets)

head(s)

tweets[4]

# Plot a barplot of the sentiment scores
barplot(colSums(s), las = 2, col = rainbow(10), ylab = 'count', main = 'Sentiments for weather')

