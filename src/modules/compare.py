import cPickle as pickle
from os import listdir

# DESCRIPTION:
    # given a date directory, return all pickle files
def drop_invalid( src_dir, date ):
    files = listdir(src_dir+date)
    job_pickles = []
    
    for f in files: 
        try:
            pickle_file = open( f, 'rb')
            jobid = pickle.load(pickle_file)
            job_pickles.append(jobid)
            pickle_file.close()
        except:
            next 
    return job_pickles
            
# DESCRIPTION:
    # given a date directory of pickle files
    # return files for jobs which ran at/above minimum
        # Note: minimum = 4  >>>  1 hour run time for job
def drop_below( src_dir, date, minimum=4 ):
    files = listdir(src_dir+date)
    job_pickles = []
    
    for f in files: 
        try:
            pickle_file = open( f, 'rb')
            jobid = pickle.load(pickle_file)
            if (len(jobid.times) > minimum):
                job_pickles.append(jobid)
                
            pickle_file.close() 
            
        except:
            next 
    return job_pickles