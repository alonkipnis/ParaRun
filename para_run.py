import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from evaluate_iteration import evaluate_iteration

import logging
logging.basicConfig(level=logging.INFO)
import argparse

from evaluate_iteration import evaluate_iteration

    
class ParaRun :

    def __init__(self, param_file='params.yaml') :
        with open(param_file) as file:
            self._params = yaml.load(file, Loader=yaml.FullLoader)
        logging.info(f" Reading parameters from {param_file}.")
        self._conf = pd.DataFrame(self._conf_generator())
        logging.info(f" {len(self._conf)} configurations generated.")

        self._out = pd.DataFrame()
        self._npartitions = 4

        
    def _conf_generator(self) :
        def gen_series(var) :
            rec = self._params['variables'][var]
            tp = self._params['variables'][var].get('float')
            return np.linspace(rec['min'], rec['max'],
                         int(rec['length'])).astype(tp)

        rr = gen_series('r')
        bb = gen_series('beta')
        xx = self._params['variables']['Zipf_params']
        nMonte = self._params['nMonte']

        nn = self._params['variables']['n_samples']
        NN = self._params['variables']['n_features']
    
        for itr in range(nMonte) :
            for N in NN :
                ee = np.round(N ** (-bb), 6)
                mm = np.round(np.sqrt(2*np.log(N) * rr), 3)
                for n in nn :
                    for eps in ee :
                        for mu in mm :
                            for xi in xx :
                                yield {'itr' : itr, 'n' : n, 'N': N,
                                       'ep' : eps, 'mu' : mu, 'xi' : xi} 
        
    def run(self, func) :
        """
        Apply atomic expriment function to each row in configuration table

        Args:
        -----
        func    atomic experiment function
        """
        logging.info(f" Running...")
        y = self._conf.iloc[:,1:].apply(lambda row : func(*row), axis=1)
        
        #self._out = pd.json_normalize(y)
        self._out = y
        logging.info(f" Completed.")

    def Dask_run(self, func) :
        """
        Apply atomic expriment function to each row in configuration table

        Args:
        -----
        func    atomic experiment function

        """

        logging.info(" Using dask.")
        import dask
        import dask.dataframe as dd
        from dask.distributed import Client

        logging.info(f" Initializing Dask client...")
        client = Client()
        logging.info(f" Client info:")
        logging.info(f"\t Services: {client.scheduler_info()['services']}")
    
        logging.info(f" Running...")

        ddf = dd.from_pandas(self._conf.iloc[:,1:], npartitions=self._npartitions)
        logging.info(" Connecting to dask server...")
        
        x = ddf.apply(lambda row : func(*row), axis=1, meta=dict)
        logging.info(" Sending futures...")
    
        y = x.compute()
        

        self._out = pd.DataFrame(y)
        self._out['func'] = str(func.__name__)
        self._out['time'] = str(datetime.now())

        logging.info(f" Completed.")
        client.close()

    def to_file(self, filename="results.csv") :
        if self._out.empty :
            logging.warning(" No output detected."
            "Did the experiment complete running?")
        if self._conf.empty :
            logging.warning(" No configuration talbe detected."
            "Did call gen_conf_table() ")

        logging.info(f" Saving results...")
        results = pd.concat([self._conf, self._out], axis=1)
        logging.info(f" Saved {len(results)} records in {filename}.")
        results.to_csv(filename)


def main() :
    parser = argparse.ArgumentParser(description='Launch experiment')
    parser.add_argument('-o', type=str, help='output file', default='results.csv')
    parser.add_argument('-p', type=str, help='yaml parameters file.', default='params.yaml')
    parser.add_argument('--dask', action='store_true')
    args = parser.parse_args()
    #
    

    exper = ParaRun(args.p)
    if args.dask :
        exper.Dask_run(evaluate_iteration)
    else :
        exper.run(evaluate_iteration)

    exper.to_file(args.o)

if __name__ == '__main__':
    main()
