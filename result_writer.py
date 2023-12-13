import json
from person import Person
'''
IMPORTANT !!!!!!!!!!!!!
Result wrtiter idea is converting list of Person objects to json file (I think that saving fetures of one person to one Person object os OK)
BUT it is converting only PUBLIC attribiutes (this non starting with _) so the main idea is that the "main" features are public and 
other are private.
TLDR: only public attributes are print to result 
'''

class ResultWriter:
    def __init__(self,res_file) -> None:
        self._res_file = res_file
        pass

    def write_ans(self,people_list) -> None:
        ans = {'people':[{feature:vars(person)[feature] for feature in vars(person) if feature[0] != '_'} for person in people_list]}
        with open(self._res_file,'w+') as f:
            json.dump(ans,f,indent=4)
