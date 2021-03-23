version = '1.0.0'

def flatten_config(y):
    out = {}
  
    def flatten(x, name =''):
          
        # If the Nested key-value 
        # pair is of dict type
        if type(x) is dict:       
            for a in x:
                flatten(x[a], name + a + '_')
        # If the Nested key-value
        # pair is of list type
        elif type(x) is list:
            items = []
            for a in x:                
                items.append(a)
            out[name[:-1]] = ",".join(items)
        else:
            out[name[:-1]] = x
  
    flatten(y)
    return out

import datetime
dt_start = datetime.date(2010,1,1)
dt_end = datetime.date(2020,1,1)
# load protected credentials
try:
    import yaml
    with open('.creds.yaml') as fh:
        cfg_yaml = yaml.safe_load(fh)
except:
    pass
