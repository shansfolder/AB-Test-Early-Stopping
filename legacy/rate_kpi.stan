data {
	int<lower=0> Nc; 		// number of entities in the control group
	int<lower=0> Nt; 		// number of entities in the treatment group
	int<lower=0> t_c[Nc];	// number of trials in the control group
	int<lower=0> t_t[Nt];	// number of trials in the treatment group
	int<lower=0> s_c[Nc]; 	// number of successes in the control group
	int<lower=0> s_t[Nt]; 	// number of successes in the treatment group
}

parameters {
	real phi_c;				// logit transform of theta in the control group
	real alpha;				// diff in phi
	real delta;				// effect size
	real<lower=0> sigma_alpha;			// distribution of alpha
}

transformed parameters {
	real mu_alpha;						// distribution of alpha
	//real mu_phi_c;
	//real<lower=0> sigma_phi_c;

	mu_alpha = delta * sigma_alpha;
}

model {
	// prior
	delta ~ cauchy(0,1);
	//mu_alpha ~ normal(0,1);
	sigma_alpha ~ normal(0,1);

	// likelihood
	//phi_c ~ normal(mu_phi_c, sigma_phi_c);
	s_c ~ binomial(t_c, inv_logit(phi_c));
	s_t ~ binomial(t_t, inv_logit(phi_c+alpha));
	alpha ~ normal(mu_alpha, sigma_alpha);
}
