# File dependencies
from tacc_stats.pickler.job_stats import Job
import cPickle as pickle
import argparse
import time as clock
from os import listdir
from IPython.display import clear_output

# Data manipulation dependencies
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# Directory of all pickled jobs via comet
source_dir = '/oasis/projects/nsf/sys200/tcooper/xsede_stats/comet_pickles/'

# List of date directories in source_dir
dates_list = [ date for date in listdir(source_dir) ]

# Total number of files to parse
n = float(sum([len(listdir(source_dir+date)) for date in dates_list]))

# Describe search contents before parsing
print "Total Files: ", int(n)

# Access and open pickled job files
#**Process:**
#    - Iterate through the non-empty date folders available in source_dir
#    - A file is saved in valid_jobs if:
#        * The pickled file is a Job object
#        * The job ran for more than 6 cycles (1 hour)
#        * The total number of jobs saved at the end of the previous date folder is less than 1000
#            _This is purely to keep the computations manageable according to compute time requested_
#    - Exceptions are skipped
    
valid_jobs = []
t0 = clock.time()
total = 0

for date in dates_list:
    
    if len(valid_jobs) > 100:
        break
    
    if len( listdir(source_dir+date) ) != 0:
        size = len(listdir(source_dir+date))
        current = 0

        for job in listdir(source_dir+date):
            total += 1
            current += 1
            clear_output(wait=True)
            print("Processing file {} of {} files for {} \t ({}% of total)"
                  .format(current, size, date, np.round( total/n*100, 2)))
            print("Files scanned: {}".format(total))
            print("Files saved: {}".format(len(valid_jobs)))
            
            # open job file if possible
            try:
                pickle_file = open( source_dir+date+'/'+job, 'rb')
                jobid = pickle.load(pickle_file)
                
                # only save jobs that ran longer than 1 hour
                if (len(jobid.times) > 4):
                    valid_jobs.append(jobid)
                    
                pickle_file.close()
                
            except:
                next 
            t2 = clock.time()
            print
            print("Run time: {}s".format(np.round(t2-t0, 1)))
            
# Format Columns & Clean Data
# Function: Given DataFrame, rename all columns with full label
# Disclaimers:
#    - Certain column descriptions are repeated exactly in the available documentation. As a result, when these columns are relabelled according to their description, a column-specific identifier is appended in parentheses to keep it unique and prevent altering the meaning unintentionally.
#    - Some of the intel categories are listed in the available documention as "-snb(hsw)-"; however, the code is actually tagged with "-hsw-". This is to note the respective categories are in fact included in this program, though they appear skipped.
#    - At least one stat type was present in the data but does not appear to have a corresponding value in the available documentation, 'intel_rapl'. This has been interpretted to represent, "Running Average Power Limit" and is included in the proceeding analysis.


# Notes:
#    - If a value is missing from the data, it will be replaced with '0' for the purpose of this project
#    - If a type of statistic was not collected on the job, that column is dropped from the DataFrame
#    - Two files are created during each iteration:
#         1) A .csv of the descriptive statistics for that host,job pair
#         2) A full .csv of the host,job data from the formatted DataFrame
#    - Naming convention: Files are labelled as '{host}_{jobid}' to support random lookup
#         * A job run on multiple host nodes is processed and saved with each individual host,job pair *

schemas = {}
schemas_devices = {}

total = 0
n = float(len(valid_jobs))
t0 = clock.time()

for job in range( len(valid_jobs) ):
    clear_output(wait=True)

    # general job values
    jobid = valid_jobs[job]
    start = pd.to_datetime(round(jobid.start_time), unit='s').time()
    end = pd.to_datetime(round(jobid.end_time), unit='s').time()
    numCycles = len(jobid.times)
    total += 1
    type_avgs = {}
    times = []
    
    ##################################
    #  build master list of schemas  #
    ##################################
    for stat in jobid.schemas.keys():
        if stat not in schemas.keys():
             schemas[stat] = jobid.schemas[stat].keys()
    
    # iterate through each host object job was run on
    for host_name, host in jobid.hosts.iteritems():
        try:
            print("Processing hosts for job {} of {} \t ({}% of total)".format(job+1, int(n), np.round( (job+1)/n*100, 2)))
            
            ##################################
            #    convert timestamps to dt    #
            ##################################
            times.append(start)
            for time in host.times:
                times.append( pd.to_datetime(round(time), unit='s').time() )
            times.append(end)
            
            ##################################
            #  build master list of devices  #
            ##################################
            for stat in host.stats.keys():
                if stat not in schemas_devices.keys():
                    schemas_devices[stat] = host.stats[stat].keys()
                 
            indices_all = []
            for stat,devices in schemas_devices.items():
                for device in devices:
                    for schema in schemas[stat]:
                        indices_all.append( (stat,device,schema) )
    
            all_idx = pd.MultiIndex.from_tuples(indices_all, names=['Stat', 'Device', 'Schema'])  
            all_df = pd.DataFrame( index=all_idx, columns=times ).sort_index()
            
            ##################################
            #   iterate through host.stats   #
            ##################################
            for host_name,host in jobid.hosts.items():
                for stat,devices in host.stats.items():
                    for device,cycles in devices.items():
                        for i in range(len(cycles)):
                            for j in range(len(cycles[i])):
                                try:
                                    time = times[i]
                                    schema = schemas[stat][j]
                                    all_df.loc[(stat,device,schema),time] = cycles[i][j]
                                except:
                                    next
                        
            #t2 = clock.time()
            #print("total: {}s".format(np.round(t2-t0, 1)))
            
            all_df.to_csv(path_or_buf="./jobs/all/{}_{}.csv".format( host_name, jobid.id ))
            
        except:
            next

# check that no job was missed
if total == len(valid_jobs):
    print "Success!"
else:
    print len(valid_jobs) - total, "jobs missing"