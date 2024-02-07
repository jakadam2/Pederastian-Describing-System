import json


class ResultWriter:

    def __init__(self,res_file) -> None:
        self._res_file = res_file
        pass

    def write_ans(self,people_list) -> None:
        ans = {'people':[{feature:vars(person)[feature] for feature in vars(person) if feature[0] != '_'} for person in people_list]}
        with open(self._res_file,'w+') as f:
            json.dump(ans,f,indent=4)
