
source("rdrobust.R")
source("rdbwselect.R")
source("functions.R")
source("rdplot.R")
source("rdbwselect_2014.R")

## Both male and female
x_b <- c(2010, 2011, 2012, 2013, 2014, 2015)
y_b <- c(109.,  93., 123., 124., 138., 115.)
y_norm_b <- c(13.75307552, 11.43081896, 14.83697422, 13.82586105, 19.00224447,
         15.57758994)
summary(rdrobust(y_b, x_b, c=2012.5, p=1, h=3))

summary(rdrobust(y_norm_b, x_b, c=2012.5, p=1, h=3))

# Male Only
x_m <- c(2010, 2011, 2012, 2013, 2014, 2015)
y_m <- c(81.00000,
         67.00000,
         90.00000,
         90.00000,
         95.00000,
         77.00000) 
y_norm_m <- c(17.83709,
       14.65987,
       19.42334,
       18.00576,
       22.24824,
       18.39465)
summary(rdrobust(y_m, x_m, c=2012.5, p=1, h=3))

summary(rdrobust(y_norm_m, x_m, c=2012.5, p=1, h=3))


# Female Only
x_f <- c(2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016)
y_f <- c(20.00000,
       28.00000,
       26.00000,
       33.00000,
       34.00000,
       43.00000,
       38.00000,
       41.00000)
y_norm_f <- c(7.04151,
         8.27325,
         7.29190,
         9.02502,
         8.56358,
         14.37022,
         11.88837,
         12.94928)
summary(rdrobust(y_f, x_f, c=2012.5, p=1, h=4))

summary(rdrobust(y_norm_f, x_f, c=2012.5, p=1, h=3))

summary(rdrobust(y_f, x_f, c=2012.5, p=2, h=4))

