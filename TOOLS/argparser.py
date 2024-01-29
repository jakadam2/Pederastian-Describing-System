import argparse

class Parser(): 

    def __init__(self): 
        parser = argparse.ArgumentParser(description='AV parser')
        parser.add_argument('--video', help='.mp4 video to process')
        parser.add_argument('--configuration', help='.txt rois configuration')
        parser.add_argument('--results', help='.txt file on wich write the results')
        self.parser = parser 

    def parse(self): 
        args = self.parser.parse_args()
        return args
    

# --------------- USAGE ----------------------

# def main(): 
#     pars = parser()
#     args = pars.parse()
#     print('video = '+args.video)
#     print('configuration = '+args.configuration)
#     print('results = '+args.results)

# if __name__ == '__main__': 
#     main()