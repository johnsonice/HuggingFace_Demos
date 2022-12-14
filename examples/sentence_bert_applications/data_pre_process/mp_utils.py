# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:10:47 2019

@author: chuang
"""

import os
import time
from multiprocessing import Pool
## multi process util object 

class Mp():
    """
    abstract base class for Generator that yields info from each doc in a dir
    :param input: File or Dir
    """
    def __init__(self, input, mp_func):
        self.input = input
        self.mp_func = mp_func

    def chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
            
    def multi_process_files(self,workers=os.cpu_count()-1,chunk_size=1000):
        print('Start multiprocessing {} files in {} cores'.format(len(self.input),workers))
        start = time.time()
        batch_size = workers*chunk_size*5
        batches = list(self.chunks(self.input, batch_size))
        p = Pool(workers)
        
        res = list()
        for i in range(len(batches)):
            print('Processing {} - {} files ...'.format(i*batch_size,(i+1)*batch_size))
            rs = p.map(self.mp_func, batches[i],chunk_size)
            res.extend(rs)
        p.close()
        p.join()
        end = time.time()            
        print(time.strftime('%H:%M:%S', time.gmtime(end - start)))

        return res

def test_func(a):
    return a+1    
#%%
if __name__ == "__main__":

    
    x = [1,2,3,4]
    
    mp = Mp(x,test_func)
    res = mp.multi_process_files(workers=1,chunk_size=1)
    print(res)