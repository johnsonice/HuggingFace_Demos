import json

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