library(rddensity)
x<-c(rep(2018,105103),rep(2017,94279),rep(2016,72721),rep(2015,73824),rep(2014,72623),rep(2013,89687),rep(2012,82901),rep(2011,81359),rep(2010,79255),rep(2009,69307),rep(2008,62904),rep(2007,79020),rep(2006,78121))
out=rddensity(X=x, c=2012.5)
summary(out)
