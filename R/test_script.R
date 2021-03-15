
source("rdrobust.R")
source("rdbwselect.R")
source("functions.R")
source("rdplot.R")
source("rdbwselect_2014.R")

x <- c(2010, 2011, 2012, 2013, 2014, 2015)
y <- c(109.,  93., 123., 124., 138., 115.)
y_n <- c(13.75307552, 11.43081896, 14.83697422, 13.82586105, 19.00224447,
         15.57758994)
summary(rdrobust(y, x, c=2012.5, p=1, h=3))

summary(rdrobust(y_n, x, c=2012.5, p=1, h=3))

