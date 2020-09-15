# System dependencies
from os import listdir
import time as clock
from datetime import timedelta
from IPython.display import clear_output

import pickle
import gzip
import re

# Data manipulation dependencies
import pandas as pd
import numpy as np
import datetime as dt

# System data locs
source_dir = '/oasis/projects/nsf/sys200/stats/xsede_stats/'
locs = { 'aofa': source_dir+'archive_of_archive',
         'job_info': source_dir+'comet_accounting',
         'arc': source_dir+'archive'
         #'host_info': source_dir+'comet_hostfile_logs',
         #'old_pickles': source_dir+'comet_pickles'
       }
acct_info_locs = [ locs['job_info']+'/'+stamp for stamp in listdir(locs['job_info']) ]
curr_data = [ locs['arc']+'/'+host_dir+'/'+stamp 
            for host_dir in listdir(locs['arc'])
            for stamp in listdir(locs['arc']+'/'+host_dir)  ]

aofa_data = [ locs['aofa']+'/'+host_dir+'/'+stamp 
            for host_dir in listdir(locs['aofa'])
            for stamp in listdir(locs['aofa']+'/'+host_dir)  ]
arc_data = curr_data + aofa_data

### Prep Cleaning

def get_time( spec=None ):
    if type(spec) is str:
        try:
            spec = float( spec )
        except:
            spec = get_stamp( spec )
    return clock.strftime("%Y-%m-%dT%H:%M:%S", clock.localtime( spec ))

def get_stamp( spec ):
    try:
        sf = "%Y-%m-%dT%H:%M:%S"
        return int(clock.mktime( clock.strptime( spec, sf ) ))
    except:
        try:
            sf = "'%Y-%m-%dT%H:%M:%S'"
            return int(clock.mktime( clock.strptime( spec, sf ) ))
        except:
            if type(spec) is int:
                return spec
            else:
                return 0

def comp_window( t_0, t_n, t_x ):
    try:
        
        # tester (t_x) comes before expected initial (t_0)
        # or after expected final (t_n)
        # NOTE: For tester.gz files, it's possible s/e window is very small and contained completely w/in file
        if ( t_x < t_0 ) or ( t_x > t_n ):
            return 1
        
        # tester within expected s/e window
        elif t_0 <= t_x <= t_n:
            return 0
        
    except:
        return -1            

def check_static( alist ):
    return alist[1:] == alist[:-1]

def check_header( line ):
    if line.find(" ") < 0:
        try:
            return line[0] == '%'
        except:
            return False
        
    else:
        chunks = line.split(" ")
        try:
            return (chunks[0][0] == '%') or ( chunks[2].find("comet") >= 0 )
        except:
            return False

def check_job( chunk ):
    return chunk.find("-") == -1

def open_txt( txt_file ):
    
    with open( txt_file, "rt" ) as f:
        lines = f.readlines()
        f.close()
    
    return lines

def unzip_txt( gzipped ):
    
    with gzip.open( gzipped, 'rt') as f:
        lines = f.readlines()
        f.close()
    
    return lines

def group_from_txt( txt_file ):
    lines = open_txt( txt_file )
    group = []
    
    for line in lines:
        chunks = line.split(" ")
        nodelist_r = chunks[0]
        nodelist = format_nodelist( chunks[0] )
        start = get_stamp( chunks[1] )
        end = get_stamp( chunks[2] )
        
        item = ( nodelist, start, end )
        group.append(item)
        
    return group

# want to split into:
#      1) from file and
#      2) from search list
def collect_ids( acct_file ):
    chunks = open_txt( acct_file )    
    test_ids = []

    for chunk in chunks[1:]:
        cut = chunk.split("|")
        test_ids.append(cut[0])
    
    return test_ids

def collect_headers( host_gzfile ):
    lines = unzip_txt( host_gzfile )
    return [ line for line in lines if 'comet' in line and ' ' in line ]

def quick_save( obj, label=get_time() ):
    
    try:
        out_file = open( label, 'wb')
        pickle.dump( obj, out_file)
        
        # double check save
        check_cpicore_set = pickle.load(open(cpiset_out, 'rb'))
        check_cpicore_set = None
        
    except:
        "There was a problem pickling the object - Save manually."

####Formatting

def format_header( line ):
    chunks = line.split(" ")
    
    try:
        if chunks[0][0] == '%':
            return {}
        else:
            return { "Timestamp": get_time( chunks[0] ), 
                     "Jobid": chunks[1],
                     "Host": chunks[2][:11] }
        
    except:
        return {}

def format_nodelist( nodelist ):
    purged = nodelist.replace('[','').replace(']','').replace(',','-').replace('-','').split("comet")[1:]
    nodes = []
    
    for item in purged:
        base = item[:2]
        prev = 2
        
        for i in range( 4,len(item)+1,2 ):
            node = 'comet' + '-' + base + '-' + item[ prev:i ]
            nodes.append(node)
            prev = i
    
    return nodes

def format_spec( line ):
    return line[1:-1]

def format_data( line ):
    chunks = line.split(" ")
    
    stat = chunks[0]
    dev = chunks[1]
    data = chunks[2:-1]
    
    return { (stat,dev): data }

def format_schema( line ):
    chunks = line.partition(" ")
    stat = chunks[0][1:]
    
    temp_sch = chunks[2:][0][:-1].replace(",E","").replace(",C","").split(" ")
    fin_sch = []
    
    for item in temp_sch:
        
        if item.find("=") > -1:
            new = item.replace(",","(") + ")"
            fin_sch.append( new )
        
        else:
            fin_sch.append( item )
    
    return { stat:fin_sch }

def separate_nodes( search_tup ):
    nl = search_tup[0]
    t_0 = search_tup[1]
    t_n = search_tup[2]
    exp_list = []
    
    if len( search_tup ) == 4:
        rem = search_tup[3]
    elif len( search_tup ) > 4:
        rem = search_tup[3:]
    
    for node in nl:
        if len(search_tup) > 3:
            exp_list.append( (node,t_0,t_n,rem) )
        else:
            exp_list.append( (node,t_0,t_n) )
    
    return exp_list
    
def from_list( chunks ):
    nl_i = chunks.index("comet")
    nl = chunks[ nl_i ]
    t_n = ''
    
    if nl_i == 0:
        t_0 = chunks[1]
    else:
        t_0 = chunks[0]
    
    try:
        for i in range(len(chunks)):
            if chunks[ i ] < t_0:
                t_0 = chunks[ i ]
            elif chunks[ i ] > t_0 and (t_n == '' or t_n < chunks[ i ]):
                t_n = chunks[ i ]                          
    except:
        next
    
    if len(chunks) > 3:
        rem = [ e for e in chunks if (e is not nl) and (e not in ts) ]
        return nl,ts,rem
    else:
        return nl,ts
    return (nl, t_0, t_n)

def from_tup( list_i ):
    nl = ''
    ts = []

    for e in list_i:
        if 'comet' in e:
            nl = e
        if 'T' in e:
            try:
                get_stamp(e)
                ts.append(e)
            except:
                next
    
    if len(list_i) > 3:
        rem = [ e for e in list_i if (e is not nl) and (e not in ts) ]
        return nl,sorted(ts),rem
    else:
        return nl,sorted(ts)

def sort_input( aline ):
    if type( aline ) is list:
        return from_list(aline)
    if type( aline ) is tuple:
        return from_tup(aline)
    
def format_search_tup( line ):
    
    if len(line) > 1:
        search_i = sort_input( line )
        
        nodelist = format_nodelist( search_i[0] )
        start = get_stamp( search_i[1][0] )
        end = get_stamp( search_i[1][1] )
        
        if len(line) == 4:
            return nodelist,start,end,line[3]
        elif len(line) > 4:
            return nodelist,start,end,line[3:]
        else:
            return nodelist,start,end
    else:
        return 1

####Data analysis

def timely_dict( host_data, host_name ):
    stamps = list(host_data[ host_name ].keys())
    schemas = host_data[ host_name ][ stamps[0] ]["Schemas"]
    timely_data = []
    
    for stamp in stamps:
        for key,data in host_data[ host_name ][ stamp ]["Data"].items():
            
            stat = key[0]
            dev = key[1]
            
            for i in range(len(data)):
                metric = schemas[stat][i]
            
            info = (stat, metric, dev, int(data[i]), stamp)
            timely_data.append( info )
    
    for i in range(len(timely_data)):
        datum_list = [ timely_data[i] ]
        t_i = timely_data[i][4]
        job_info = host_data[ host_name ][ t_i ]["Job"]
        
        try:
            if "Jobid" in job_info.keys():
                jobid = (job_info["Jobid"])
                new_tup = tuple( datum_list.append( jobid ) )
                timely_data[i] = new_tup
        except:
            continue
    
    return timely_data

####Data Munging

def info_dict( rules, info ):
    rules_list = rules.split("|")
    
    if len(rules_list) != len(info):
        return {}
    
    else:
        return { rules_list[i]:info[i] for i in range(len(rules_list)) }

def host_to_info_dict( zip_txt ):
    contents = unzip_txt( zip_txt )
    host_name = contents[1].partition(" ")[2][:11]
    out_dict = { host_name: {} }
    host_info = {}
    info_dict = { "Data":{},
                    "Job":"N/A",
                    "Schemas":{},
                    "Specs":[]
                }
    
    for line in contents:
            
        if line[0] == "$":
            info_dict["Specs"].append( format_spec( line ) )
            
        elif line[0] == "!":
            info_dict["Schemas"].update( format_schema( line ) )
        
        else:
            
            if (len(line) > 0) and (len(line) < 3 or check_header( line )):
                header_dict = format_header( line )
                
                if header_dict:
                    t = header_dict["Timestamp"]
                    host_info[ t ] = {}
                    
                    # Collecting as dictionary to support additional accounting data
                    if check_job( header_dict["Jobid"] ):
                        temp_jobid = header_dict["Jobid"]
                        info_dict["Job"] = { "Jobid": temp_jobid } 
                    
            else:
                incoming = format_data( line )
                info_dict["Data"].update( incoming )
                
                host_info[t].update( info_dict )
                
    out_dict[host_name].update( host_info )
    
    for host_name,host_data in out_dict.items():
        out_dict[ host_name ][ "Timely Data" ] = timely_dict( out_dict, host_name )
    
    return out_dict

def job_to_info_dict( txt_file_list ):
    nodes_by_date = {}
    unsaved = []

    for date in txt_file_list:
        try:
            # skip alt files
            #check_stamp = int( date[-14] )
            
            # read in file contents
            contents = open_txt( date )
            
            # formatting
            label = date[-14:-4]
            rules = contents[0]
            jobs = contents[1:]
            
            # template to save
            nodes_by_date[ label ] = {}
            nodes_by_date[ label ]["multiple"] = {}
            nodes_by_date[ label ]["rules"] = rules
            
            # run through lines in file
            for job in jobs:
                line = job.split("|")
                node = line[-1]
                info = info_dict( rules, line )
                
                # save multiple node jobs to specified loc
                if len(node) > 12:
                    nodes = format_nodelist( info )
                    for node in nodes:
                        nodes_by_date[ label ][ "multiple" ][ node ] = info
                
                else:
                    nodes_by_date[ label ][ node[:11] ] = info
        except:
            unsaved.append(date)
            
    
    return nodes_by_date, unsaved

def lookup_files( searchable_list ):
    found = []
    
    for key in searchable_list:
        host = key[0]
        t_0 = key[1]
        t_n = key[2]
        d_0 = get_time(t_0)[:10]
        d_n = get_time(t_n)[:10]
    
        for i in range(len(arc_data)):
            loc = arc_data[i]
        
            if (host in loc) and t_0 != 0:
                if str(t_0) in loc or str(t_0)[:-2] in loc:
                    if loc not in found: found.append(loc)
            
            if (host in loc) and t_n != 0:
                if str(t_n) in loc or str(t_n)[:-2] in loc:
                    if loc not in found: found.append(loc)
                    
        for i in range(len(aofa_data)):
            loc = aofa_data[i]
        
            if (host in loc) and t_0 != 0:
                if str(t_0) in loc or str(t_0)[:-2] in loc:
                    if loc not in found: found.append(loc)
            
            if (host in loc) and t_n != 0:
                if str(t_n) in loc or str(t_n)[:-2] in loc:
                    if loc not in found: found.append(loc)
         
        for i in range(len(acct_info_locs)):
            loc = acct_info_locs[i]
            
            if (d_0 in loc) or (d_n in loc):
                if loc not in found: found.append(loc)

    lost = [ e for e in searchable_list if e not in found ]
    
    return found,lost

def deep_search_acct( file_list, search_list ):
    jobids = [ item[3] for item in search_list ]
    collected = []
    
    for i in range(len(file_list)):
        possible = open_txt( file_list[i] )
        
        for jobid in jobids:
            for chunk in possible:
                if jobid in chunk:
                    collected.append(chunk)
    
    return collected

def deep_search_host( search_list ):
    out = { item:0 for item in search_list }
    
    for i in range(len( arc_data )):
        for j in range(len( search_list )):
            filename = arc_data[i]
            target = search_list[j]
            
            if target[0] in filename:
                try:
                    if comp_window( target[1], target[2], filename[-13:-3] ):
                        out[ target ] = filename
                    else:
                        headers = collect_headers( filename )
                        out[ target ] = { "Source":[], "Headers":[] }
                        
                        for k in range(len( headers )):
                            sample = headers[k].split(" ")
                            
                            # if jobids match
                            if sample[1] == target[3]:
                                
                                if filename not in out[ target ][ "Source" ]:
                                    out[ target ][ "Source" ].append( filename )
                                out[ target ]["Headers"].append(sample)
                            
                            # if timestamps match
                            elif comp_window( target[1], target[2], sample[0] ) == 0:
                                
                                if filename not in out[ target ][ "Source" ]:
                                    out[ target ][ "Source" ].append( filename )
                                out[ target ]["Headers"].append(sample)
                            
                except:   
                    continue
                    
    return out

# PARAMETERS:
# 's/e' single search from start/end (manual)
#       ie) "Start, End: 2020-01-03T20:34:47, 2020-01-05T08:15:18"
# 's' single search from nodelist%start%end (manual)
#       ie) "NL, Start, End: comet-05-12 2020-03-03T20:34:47 2020-03-05T08:15:18"
#       ie) "NL, Start, End: comet-05-[12,16] 2020-03-03T20:34:47 2020-03-05T08:15:18"
# 'l' repeated search from nodelist%start%end strings or (nodelist,start,end) tuples (from list)
#       ie) myJobList = [ "comet-05-12 2020-03-03T20:34:47 2020-03-05T08:15:18",
#                          (comet-05-12, 2020-03-03T20:34:47, 2020-03-05T08:15:18)   ]
#           search( mode='l', myJobList )
# 'f' repeated search from nodelist%start%end (from file)
#       ie) "Text file: your_search_file.txt"  (Note: Mismatched file contents ignored)
def search( mode=['s/e', 's', 'l','f'], from_list=False, ret_form=False ):
    
    if mode == 's/e':
        t_0,t_n = input("Start, End:").replace(",", "").split(" ")
        start = get_stamp( t_0 )
        end=get_stamp( t_n )
        return start,end
    
    elif mode == 's':
        line = input("NL, Start, End:").replace(",", "").split(" ")
        line_tup = format_search_tup( line )
        return line_tup

    elif mode == 'l' and type( from_list ) is list:
        out_list = []
        dropped = []
            
        for obj in from_list:
            try:
                obj_tup = format_search_tup( obj )
                
                if len( obj_tup[0] ) > 1:
                    exp_tups = separate_nodes( obj_tup )
                    out_list += exp_tups
                    
                elif len( obj_tup[0] ) == 1:
                    out_list.append( obj_tup[0][0], obj_tup[1:] )
            except:
                try:
                    if ('comet' in obj[0]) and (len(obj[0]) == 11):
                        out_list.append( obj )
                except:
                    continue
                    
        files,notFound = lookup_files( out_list )
        
        #if ret_form:
            #
        #else:
        return { "Acct Info": deep_search_acct( files, from_list ), "Host Info": deep_search_host( out_list ) }
    
    elif mode == 'f':
        search_list = group_from_text( input("Text file:") )
        return search_list
    
    else:
        return 1