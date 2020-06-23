#==============================================================================================
#Movie Recommendation System
#Pablo Dinamarca
#23/06/2020
#==============================================================================================
# NOTE: The following code is essentially the same as in the Rmarkdown document.
#==============================================================================================
#~~~~~~~~~~~~~~~~~~#
# Data Preparation #
#~~~~~~~~~~~~~~~~~~#

#-----------------------------------#
# Create edx set and validation set #
#-----------------------------------#

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(plotly)) install.packages("plotly", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)


rm(dl, ratings, movies, test_index, temp, movielens, removed)
#==============================================================================================
#-----------------------------------#
# Create the test set and train set #
#-----------------------------------#

# Then we divide the "edx" set into two subsets used for algorithm training and testing:
# 1. The "train" set with 90% of the "edx" data.
# 2. The "test" set with 10% of the "edx" data.
# The model is created and trained in the "train" set and tested in the "test" set until 
# the `RMSE` goal is reached.

library(tidyverse)
library(lubridate)
library(plotly)
library(caret)
set.seed(1, sample.kind = "Rounding")

index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
test_set <- edx[index,] 
train_set <- edx[-index,]
rm(index)

# Make sure userId and movieId in test set are also in train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Verify the data is correct
head(train_set)
#==============================================================================================
#-----------------#
# Modify the data #
#-----------------#

# If we look at the dataset, we can notice that it contains a column called "timestamp" 
# from which we can extract the years to later perform the exploration with timelines.

## Modify the year as a column
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
train_set <- train_set %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
test_set <- test_set %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
#==============================================================================================
#~~~~~~~~~~~~~~~~~~#
# Data Exploration #
#~~~~~~~~~~~~~~~~~~#

# Pre-visualize the Data
names(edx)
head(edx, 5)
str(edx)
summary(edx)
dim(edx)
class(edx)
edx %>% summarize(Users = n_distinct(userId), 
                  Movies = n_distinct(movieId))

#==============================================================================================
#-----------------------------#
# Exploration by each feature #
#-----------------------------#

# Rating

# The following plot shows that users tend to rate movies between 3 and 4. 
# This may be due to different factors and trying to generalize this observation may be wrong
# so we are going to analyze other variables.

# Note:This process could take a couple of minutes.
R <- qplot(as.factor(as.vector(edx$rating))) +
  ggtitle("Ratings Distribution") + xlab("Rating")
ggplotly(R)


#-----------------------------------------------------------------------------------------------#
# Date

# The following table lists the highest rated days for each movie. 
# We note that movies that are box office hits have higher ratings as they are better known.

edx %>% mutate(date = date(as_datetime(timestamp))) %>%
  group_by(date, title) %>%
  summarise(Count = n()) %>%
  arrange(-Count) %>%
  head(10)

D <- edx %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth()
ggplotly(D)

# We notice that people have lowered their grade point average over the years and 
# have become more critical when it comes to watching movies.


#-----------------------------------------------------------------------------------------------#
# Movie

# The histogram shows that some movies have been rated very rarely. 
# Therefore, they should be given less importance in movie prediction.

M <- edx %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30) + 
  scale_x_log10() + 
  ggtitle("Movies Distribution") +
  xlab("Number of Ratings") +
  ylab("Number of Movies")
ggplotly(M)


#-----------------------------------------------------------------------------------------------#
# User

# The graph above shows that not all users are equally active. 
# Some users have rated very few movies.

U <- edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30) + 
  scale_x_log10() + 
  ggtitle("User Distribution") +
  xlab("Number of Ratings") +
  ylab("Number of User")
ggplotly(U)


#-----------------------------------------------------------------------------------------------#
# Genres

# The Movielens dataset contains different combinations of genres. 
# Here is the list of the total movie ratings by genre.

edx %>%
  group_by(genres) %>%
  summarize(N = n()) %>%
  arrange(-(N)) %>% head(5)

#==============================================================================================
#~~~~~~~~~~~~~~~#
# Data Cleaning #
#~~~~~~~~~~~~~~~#

# Remove extra variable
rm(G,M, U, R, D)

# Remove extra Column
edx <- edx[,-4]
validation <- validation[,-4]
train_set <- train_set[,-4]
test_set <- test_set[,-4]

#==============================================================================================
#~~~~~~~~~~~~~~~~~~~~#
# Evaluation Results #
#~~~~~~~~~~~~~~~~~~~~#

# Here we define the loss functions:

# Root Mean Squared Error - RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
#==============================================================================================
#~~~~~~~~~~~~~~~~~~~#
# Create the Models #
#~~~~~~~~~~~~~~~~~~~#

#------------------#
# Model by Average #
#------------------#

# Average of Ratings
mu_hat <- mean(train_set$rating)

# Create the error dataframe
Model_Results <- data_frame(Method = "By Average", RMSE = RMSE(test_set$rating, mu_hat))

# Show the RMSE by average method
data.frame(Model_Results)
#==============================================================================================
#-----------------------#
# Model by Movie Effect #
#-----------------------#

#The Bias of movie is represent by b_i
Movie_effect <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))
head(Movie_effect, 5)

#Plot of Movie Effect
Movie_effect %>% ggplot(aes(b_i)) + 
  geom_histogram(bins=10, col = I("black")) +
  ggtitle("Movie Effect Distribution") +
  xlab("Movie effect") +
  ylab("Count")

# We can observe the ranges of the values and their distribution of b_i.

#Predict the model by this effect
Predicted_b <- mu_hat + test_set %>% 
  left_join(Movie_effect, by="movieId") %>%
  .$b_i

#Aggregate to Models Dataframe
Method_bi <- RMSE(test_set$rating, Predicted_b)
Model_Results <- bind_rows(Model_Results,
                           data_frame(Method="By Movie Effect",
                                      RMSE = Method_bi ))


data.frame(Model_Results)
#==============================================================================================
#------------------------------#
# Model by Movie & User Effect #
#------------------------------#

#The Bias of user is represent by b_u
User_effect <- test_set %>% 
  left_join(Movie_effect, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

#Predict the model by Movie and User effect
Predicted_b <- test_set %>% 
  left_join(Movie_effect, by="movieId") %>%
  left_join(User_effect, by="userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred
Method_MUE <- RMSE(test_set$rating, Predicted_b)


#Aggregate to Models Dataframe
Model_Results <- bind_rows(Model_Results,
                           data_frame(Method="By Movie & User Effect",  
                                      RMSE = Method_MUE ))
Model_Results
#==============================================================================================
#------------------------------------------#
# Model by Regularized Movie + User Effect #
#------------------------------------------#

#Select lambda by cross-validation
lambda <- seq(0, 10, 0.25)
set.seed(1, sample.kind = "Rounding")

# Note: this process could take a couple of minutes
Errors <- sapply(lambda, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  Predicted_b <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(test_set$rating, Predicted_b))
})

# Select the optimal lambda
qplot(lambda, Errors) 
lambda_1 <- lambda[which.min(Errors)]

#Save the results
Model_Results <- bind_rows(Model_Results,
                           data_frame(Method="Reg. Movie + User Effect",  
                                      RMSE = min(Errors)))
data.frame(Model_Results)
#==============================================================================================
#---------------------------------------------------#
# Model Aggregate Regularized Year and Genres effect #
#---------------------------------------------------#

# Select lambda by cross-validation
lambda <- seq(0, 20, 1)
set.seed(1, sample.kind = "Rounding")

# Note: this process could take a couple of minutes
Errors <- sapply(lambda, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_y <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by = "year") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u - b_y)/(n()+l))
  
  Predicted_b <- test_set %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>% 
    .$pred
  
  return(RMSE(test_set$rating, Predicted_b))
})

# Select the optimal lambda
qplot(lambda, Errors) 
lambda_2 <- lambda[which.min(Errors)]

# Save the result
Model_Results <- bind_rows(Model_Results,
                           data_frame(Method="Agg. Reg. Year + Genres Effect",  
                                      RMSE = min(Errors)))
data.frame(Model_Results)

# We can see that the error with the train and test data decreased significantly and 
# achieved our goal of obtaining an RMSE < 0.8649 so we are now ready to implement the algorithm 
# with the Validation set.


#==============================================================================================
#~~~~~~~~~~~~~~~~~~#
# Remove variables #
#~~~~~~~~~~~~~~~~~~#

# Remove all unnecessary variables to assess the final model with Validation set.

rm(Model_Results, Movie_effect, Rating_plot, User_effect, Method_MUE, 
   Method_bi, Method_bu, Predicted_MUE, Predicted_ME, Predicted_UE, 
   train_set, test_set, Errors, lambda, mu_hat, Predicted_b)
#==============================================================================================
#~~~~~~~~~~~~~~~~~~#
# Final Validation #
#~~~~~~~~~~~~~~~~~~#


# On this occasion we simply apply three models:
# 1. The average to have a reference.
# 2. Those regularized by User and Film.
# 3. The regularized ones including by Year and Gender.

# This is due to ease in computing and with previous knowledge obtained from previous models.
#==============================================================================================
#------------------#
# Model by Average #
#------------------#

set.seed(1, sample.kind = "Rounding")

Model_Results <- data_frame(Method = "By Average", 
                            RMSE = RMSE(validation$rating, mean(edx$rating)))
data.frame(Model_Results)
#==============================================================================================
#------------------------------------------#
# Model by Regularized Movie + User Effect #
#------------------------------------------#

# Select lambda_1 of test_set
set.seed(1, sample.kind = "Rounding")

# Note:This process could take a couple of minutes
Errors <- sapply(lambda_1, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  Predicted_B <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(validation$rating, Predicted_B))
})

#Save the results
Model_Results <- bind_rows(Model_Results,
                           data_frame(Method="Reg. Movie + User Effect",  
                                      RMSE = min(Errors)))
data.frame(Model_Results)
#==============================================================================================
#---------------------------------------------------#
# Model Aggregate Regularized Year and Genres effect #
#---------------------------------------------------#

# Select lambda_2 of test_set
set.seed(1, sample.kind = "Rounding")

# Note: this process could take a couple of minutes
Errors <- sapply(lambda_2, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_y <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  
  b_g <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_y, by = 'year') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u - b_y)/(n()+l))
  
  Predicted_B <- validation %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_y, by = 'year') %>%
    left_join(b_g, by = 'genres') %>%
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>% 
    .$pred
  
  return(RMSE(validation$rating, Predicted_B))
})

# Save the result
Model_Results <- bind_rows(Model_Results,
                           data_frame(Method="Agg. Reg. Year + Genres Effect",  
                                      RMSE = min(Errors)))
data.frame(Model_Results)
#==============================================================================================
#           Method	                   RMSE
# By Average                    	  1.0612018			
# Reg. Movie+User Effect	          0.8648177			
# Agg. Reg. Year+Genres Effect	    0.8642529	
#==============================================================================================
#~~~~~~~~~~~~~~#
# Final Report #
#~~~~~~~~~~~~~~#

# We can see that the error is 0.8643  < 0.8649. 
# So we managed to meet the objective of this project and created an effective predictive model 
# for movie recommendation systems.

