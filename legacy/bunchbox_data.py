import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import csv
from optparse import OptionParser

import sys
sys.path.insert(0, '/home/jbao/project/github/expan')
from expan.core.experiment import Experiment
metadata = {'source': 'bunchbox', 'experiment': 'example'}

params = {
	'5609412b5c6ee5f32c12f85d':{
		'start':1445904000000,
		'end':1448323200000,
		'baseline':'562df9aa50bd667b63143313',
		'variant':'562df9aa50bd667b63143312'},
	'56b8a44026f9d8072e54a73d':{
		'start':1454976000000,
		'end':1458000000000,
		'baseline':'56b8a44026f9d8072e54a747',
		'variant':'56b8a44026f9d8072e54a746'}	
}

def load_experiment(eid):
	dat = pd.read_csv('data/participations_'+eid+'.csv')
	# filter for full week cycles	
	dat = dat[(dat.timestamp>=params[eid]['start']) & (dat.timestamp<=params[eid]['end'])]
	# time difference in days
	dat.loc[:,'time_since_start'] = (dat.timestamp - min(dat.timestamp)) / 1000. / 60 / 60 / 24
	# rename
	dat.rename(columns={'participationId':'entity', 'variantId':'variant'}, inplace=True)

	return dat


def fixed_horizon(eid):
	dat = load_experiment(eid)
	snapshot = dat[dat.time_since_start<100]
	#kpi = snapshot.groupby(['entity','variant']).converted.sum().reset_index()
	kpi = snapshot.groupby(['entity','variant']).converted.mean().reset_index()
	exp = Experiment(params[eid]['baseline'], kpi, metadata)
	res = exp.delta(kpi_subset=['converted'])

	return res


def early_stopping(eid, method, day_index):
	dat = load_experiment(eid)
	max_sample_size = float(len(np.unique(dat.entity)))
	print(max_sample_size)
	metadata['estimatedSampleSize'] = max_sample_size
	# daily peeking
	#for day in np.arange(1,np.ceil(max(dat.time_since_start))+1):
	snapshot = dat[dat.time_since_start<day_index]
	
	# sum
	#kpi = snapshot.groupby(['entity','variant']).converted.sum().reset_index()
	# mean
	kpi = snapshot.groupby(['entity','variant']).converted.mean().reset_index()
	
	current_sample_size = kpi.shape[0]
	exp = Experiment(params[eid]['baseline'], kpi, metadata)
	#res = exp.delta(method='group_sequential', kpi_subset=['converted'],
	#	information_fraction=current_sample_size/max_sample_size)
	if 'bayes' in method:
		res = exp.delta(method=method, kpi_subset=['converted'],
			distribution='normal')
	elif method == 'group_sequential':
		res = exp.delta(method='group_sequential', kpi_subset=['converted'],
			information_fraction=current_sample_size/max_sample_size)
	else:
		raise NotImplementedError
	
	return (day_index, res.statistic('delta', 'stop', 'converted').loc[:,('value',params[eid]['variant'])].values[0], res.statistic('delta', 'uplift', 'converted').loc[:,('value',params[eid]['variant'])].values[0])


def run(cpus, eid, method):
	dat = load_experiment(eid)
	results = Parallel(n_jobs=int(cpus)) (
                delayed(early_stopping)(eid, method, d) for d in np.arange(1,np.ceil(max(dat.time_since_start))+1)
            )
	filename = 'bunchbox/' + eid + '_' + method + '.csv'

	with open(filename, 'w') as file_handler:
		out = csv.writer(file_handler)
		for item in results:
			out.writerow(item)


if __name__ == '__main__':
	#eid = '5609412b5c6ee5f32c12f85d'
	#eid = '56b8a44026f9d8072e54a73d'	# a/a test
	#interval = res.statistic('delta', 'uplift_pctile', 'normal_same').loc[:,('value','B')]
	parser = OptionParser()
	parser.add_option("-c", "--cpu", dest="cpu", 
            help="Number of CPUs to use, default=1", default=1, type='int')
	parser.add_option("-e", "--eid", dest="eid", 
            help="Experiment Id", type='str')
	parser.add_option("-m", "--method", dest="method", 
            help="Early stopping method", type='str')
	(options, args) = parser.parse_args()
	run(options.cpu, options.eid, options.method)
