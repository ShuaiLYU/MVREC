
import json
def load_json(json_file):
    with open(json_file, encoding='utf-8') as f:
        data=json.load(f)
        return data

def save_json(json_data,json_path):
        with open(json_path, "w",encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

def is_json(fname):
    if fname.endswith(".json"):
        return True
    return False




class JsonData(object):
    
    def __init__(self,json):
        
        assert(isinstance(json,(dict,str)))

        if isinstance(json,str):
            self.json_data=load_json(json)
        else:
            self.json_data=json
    
    def save_to(self,json_path):
        save_json(self.json_data,json_path)
            
