library(dplyr)
library(rstan)
library(doMC)

# Read command-line arguments
args=(commandArgs(TRUE))
# for (i in 1:length(args)) {
#    eval(parse(text=args[[i]]))
# }
cpus <- as.numeric(args[1])
func_name <- args[2]
kpi <- args[3]
model_file <- args[4]
rope_width <- as.numeric(args[5])
hdi_mass <- as.numeric(args[6])

registerDoMC(cpus)

#setwd('~/project/ghe/early-stopping')
sims <- 100
total_entities <- 20000
daily_entities <- 2000
days <- 20

generate_random_data <- function(seed) {

	set.seed(seed)

	assignment <- data.frame(entity=1:total_entities,
							             variant=sample(c('A', 'B'), size=total_entities, replace=TRUE))

	all_data <- NULL
	for (d in 1:days) {
		test_data_frame <- data.frame(entity=sample(1:total_entities, size=daily_entities, replace=FALSE))
		test_data_frame <- merge(test_data_frame, assignment, by='entity')

		#test_data_frame$normal_same <- np.random.normal(size=daily_entities)
		#test_data_frame['normal_shifted'] = np.random.normal(size=daily_entities, scale=10)
		#test_data_frame.loc[test_data_frame['variant'] == 'B', 'normal_shifted'] \
		#	= np.random.normal(loc=0.4, size=test_data_frame['normal_shifted'][test_data_frame['variant'] == 'B'].shape[0])

		theta = 0.75
		#test_data_frame['binomial_same'] = np.random.binomial(10, theta, size=daily_entities)/10.
		#test_data_frame['binomial_shifted'] = np.random.binomial(10, theta, size=daily_entities)/10.
		#test_data_frame.loc[test_data_frame['variant'] == 'B', 'binomial_shifted'] \
		#	= np.random.binomial(10, theta+0.1, size=test_data_frame['binomial_shifted'][test_data_frame['variant'] == 'B'].shape[0])/10.

		test_data_frame$visits <- stats::rpois(daily_entities,3)
		test_data_frame$orders_same <- stats::rbinom(length(test_data_frame$visits), test_data_frame$visits, theta)
		test_data_frame$orders_shifted <- stats::rbinom(length(test_data_frame$visits), test_data_frame$visits, theta)
		test_data_frame$orders_shifted[test_data_frame$variant == 'B'] <- stats::rbinom(length(test_data_frame$visits[test_data_frame$variant == 'B']), test_data_frame$visits[test_data_frame$variant == 'B'], theta+0.1)

		test_data_frame$treatment_start_time <- d
		all_data <- rbind(all_data, test_data_frame)
	}
	#print('Finished generating data!')
	all_data
}
		
#dat = pd.read_csv('size39_sales_kpi_over_time.csv')
#dat = generate_random_data()

get_snapshot <- function(dat, start_time) {
	snapshot <- dat[dat$treatment_start_time<=start_time,]
	aggregated <- snapshot %>% group_by(entity,variant) %>% summarise_each(funs(sum))
	aggregated
}
	
HDI_from_MCMC <- function(posterior_samples, credible_mass=0.95) {
    # Computes highest density interval from a sample of representative values,
    # estimated as the shortest credible interval
    # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
    # http://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    sorted_points <- order(posterior_samples)
    ciIdxInc <- ceiling(credible_mass * length(sorted_points))
    nCIs <- length(sorted_points) - ciIdxInc
    ciWidth <- rep(0,nCIs)
    for (i in 1:nCIs) {
        ciWidth[i] <- sorted_points[i + ciIdxInc] - sorted_points[i]
    }
    HDImin <- sorted_points[which.min(ciWidth)]
    HDImax <- sorted_points[which.min(ciWidth)+ciIdxInc]
    list(HDImin, HDImax)
}
  
# perform the fit
fit_stan <- function(sm, fit_data) {
	# fit_data = {'Nc': sum(df.variant=='A'), 
	# 			'Nt': sum(df.variant=='B'), 
	# 			'x': df[kpi][df.variant=='A'], 
	# 			'y': df[kpi][df.variant=='B']}
	fit <- sampling(sm, data=fit_data, iter=25000, chains=4)

	# extract the traces
	traces <- extract(fit)
	delta_trace <- traces$delta

	list(fit, delta_trace)
}
	
# def bayes_factor(sm, simulation_index, day_index, kpi):
# 	"""
# 	Args:
# 		sm (pystan.model.StanModel): precompiled Stan model object
# 		simulation_index (int): random seed used for the simulation
# 		day_index (int): time step of the peeking
# 		kpi (str): KPI name
# 
# 	Returns:
# 		Bayes factor based on the Savage-Dickey density ratio
# 	"""
# 	dat = generate_random_data(simulation_index)
# 	df = get_snapshot(dat, day_index+1)
# 	fit, delta_trace = fit_stan(sm, df, kpi)
# 	kde = gaussian_kde(delta_trace)
# 	return kde.evaluate([0]) / cauchy.pdf(0,loc=0,scale=1)

hdi_rope <- function(sm, simulation_index, day_index, kpi, rope_width, hdi_mass=0.95) {
	dat <- generate_random_data(simulation_index)
	df <- get_snapshot(dat, day_index)
	fit_data <- get_fit_data(df, kpi)
	tmp <- fit_stan(sm, fit_data)
	delta_trace <- tmp[[2]]
	hdi <- HDI_from_MCMC(delta_trace, hdi_mass)

	rope_lower <- - rope_width / 2
	rope_upper <- rope_width / 2

	if (hdi[[1]]>=rope_lower & hdi[[2]]<=rope_upper) {
		return('inside')
	} else if (hdi[[2]]<=rope_lower | hdi[[1]]>=rope_upper) {
		return('outside')
	} else {
		return('overlap')
	}
}

get_fit_data <- function(df, kpi) {
	fit_data <- switch(kpi,
					   normal_shifted = list(Nc=sum(df$variant=='A'), 
		                 					 Nt=sum(df$variant=='B'), 
		                 					 Vc=df$normal_shifted[df$variant=='A'], 
		                 					 Vt=df$normal_shifted[df$variant=='B']),
					   binomial_shifted=list(Nc=sum(df$variant=='A'), 
		                 					 Nt=sum(df$variant=='B'), 
		                 					 Vc=df$visits[df$variant=='A'], 
		                 					 Vt=df$visits[df$variant=='B'],
		                 					 Oc=df$orders_shifted[df$variant=='A'], 
		                 					 Ot=df$orders_shifted[df$variant=='B'])
					   )
	fit_data
}

func <- eval(parse(text=func_name))
sm <- stan_model(file=model_file)
res <- foreach(si=1:sims, .combine='c') %:%
		foreach(di=1:days, .combine='c') %dopar% {	
  			func(sm, si, di, kpi, rope_width, hdi_mass)
		}
write.table(res, paste0(func_name,'/',func_name,'_',kpi,'_',rope_width,'_',hdi_mass,'.csv'), quote=FALSE, row.names = FALSE, col.names = FALSE)