import json
import itertools

def load_json(f_path):
    with open(f_path) as f:
        data = json.load(f)
    
    return data

def load_jsonl(fn):
    result = []
    with open(fn, 'r') as f:
        for line in f:
            data = json.loads(line)
            result.append(data)
    return result 

def to_jsonl(fn,data,mode='w'):
    with open(fn, mode) as outfile:
        if isinstance(data,list):
            for entry in data:
                json.dump(entry, outfile)
                outfile.write('\n')
        else:
            json.dump(data, outfile)
            outfile.write('\n')
            

def hyper_param_permutation(hyper_params):
    ## prepare params for gride search ##
    assert isinstance(hyper_params,dict)
    param_names = list(hyper_params.keys())
    all_param_list = [hyper_params[k] for k in param_names]
    permu = list(itertools.product(*all_param_list))
    res = [dict(zip(param_names,i)) for i in permu]
    return res


if __name__ == "__main__":
    
    
    inp_param = {'batch':[1,2,3],
                 'lr':[0.1,0.2,0.3]}
    
    res = hyper_param_permutation(inp_param)
    print(res)