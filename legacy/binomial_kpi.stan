data {
	int<lower=0> Nc; 	// number of entities in the control group
	int<lower=0> Nt; 	// number of entities in the treatment group
	int<lower=0> Vc[Nc]; 		// visits in the control group
	int<lower=0> Vt[Nt]; 		// visits in the treatment group
	int<lower=0> Oc[Nc]; 		// orders in the control group
	int<lower=0> Ot[Nt]; 		// orders in the treatment group
}

parameters {
	real<lower=0,upper=1> theta_c;
	real<lower=0,upper=1> theta_t;
}

transformed parameters {
	real delta;
	delta = theta_t - theta_c;
}

model {
	delta ~ cauchy(0, 1);
	Oc ~ binomial(Vc, theta_c);
	Ot ~ binomial(Vt, theta_t);
}

