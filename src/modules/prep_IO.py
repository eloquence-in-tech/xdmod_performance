# System dependencies
from os import listdir
import time as clock
from datetime import timedelta
from IPython.display import clear_output

import pickle
import gzip

# Data manipulation dependencies
#import pandas as pd
import numpy as np
import datetime as dt
#from collections import OrderedDict
import scipy
import statistics

# Custom dependencies
import prep_fs

acct_info_locs = prep_fs.acct_info_locs
arc_data = prep_fs.arc_data
sorted_arc_data = prep_fs.sorted_arc_data

conv = prep_fs.conv_acct

### Prep Analysis
def check_static( alist ):
    return alist[1:] == alist[:-1]

def fresh_start( host_data_list ):
    return (host_data_list[0] == 0) and ( all(i <= j for i, j in zip(host_data_list, host_data_list[1:]) ))

def monotonic( nice_list ):
    dx = np.diff( nice_list )
    return np.all(dx <= 0) or np.all(dx >= 0)

def get_stats( data ):   
    try:
        stats_blob = {
            'Count' : len(data),
            'Min' : min(data),
            'Max' : max(data),
            'Mode' : statistics.mode(data),
            'Quartiles' : np.percentile(data, [25, 50, 75] ),
            'Mean' : np.mean(data),
            'Std. Dev' : np.std(data),
            'Skew' : scipy.stats.skew(data),
            'Values': data
        }
        return stats_blob
    
    except:
        stats_blob = {
            'Count' : len(data),
            'Min' : min(data),
            'Max' : max(data),
            'Quartiles' : np.percentile(data, [25, 50, 75] ),
            'Mean' : np.mean(data),
            'Std. Dev' : np.std(data),
            'Values': data
        }
        return stats_blob
    
def get_cov_matrix( x_list, y_list ):
    x_arr,y_arr = np.array( x_list ), np.array( y_list )
    
    return np.cov( x_arr, y_arr )

def get_corr_matrix( x_list, y_list ):
    x_arr,y_arr = np.array( x_list ), np.array( y_list )
    
    return np.corrcoef( x_arr, y_arr ) #r = corrcoef_matrix[0, 1]

def corr_w_pval( x_list, y_list ):
    x_arr,y_arr = np.array( x_list ), np.array( y_list )

    return scipy.stats.pearsonr( x_arr, y_arr )

def linear_reg( x_list, y_list ):
    x_arr,y_arr = np.array( x_list ), np.array( y_list )

    #LinregressResult has .slope, .intercept, .rvalue, .pvalue, .stderr
    return scipy.stats.linregress( x_arr, y_arr )

def norm_df( df ):
    return (df-df.mean())/df.std()

def minmax_df( df ):
    return (df-df.min())/(df.max()-df.min())

def get_sample_hosts( search_set, n ):
    out = []
    
    for i in range(n):
        idx = np.random.randint( len(search_set) )
        val = search_set[ idx ]
        
        if '_' not in val[3]:
            out.append( val )
        else:
            idx = np.random.randint( len(search_set) )
            val = search_set[ idx ]
            out.append( val )
        
    return out

def get_host_descriptives( labeled_data_dict, data_keys_list ):
    descriptives = { "Static":[] }
    
    for i in range(len( data_keys_list )):
        key = data_keys_list[i]
        timedata_list = [ float(item[1]) for item in labeled_data_dict[ key ] ]
        
        if ( sum(timedata_list) == 0 ) or ( check_static( timedata_list ) ):
            descriptives["Static"].append( key )
            
        else:
            
            if not fresh_start( timedata_list ):
                temp = timedata_list
                base = temp[0]
                timedata_list = [ x - base for x in temp ]
        
                stats = get_stats( timedata_list )
                stats["Nonzero Start"] = True
                stats["Starting Value"] = base
                stats["Raw Values"] = temp
                descriptives[ key ] = stats
                
            else:
                stats = get_stats( timedata_list )
                descriptives[ key ] = stats
                
    return descriptives

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

def check_header( line ):
    try: return (line.split(" ")[0] not in ['$','!','%begin','%end']) and ('comet' in line.split(" ")[2])
    except: return False

def check_job( chunk ):
    return chunk.find("-") == -1

def check_host_file( headers_list, search_tup ):
    ids = [ item.split(" ")[1] for item in headers_list ]
    target_id = search_tup[3]
    
    if (target_id in ids) and (ids[1:] == ids[:-1]):
        return 2
    elif (target_id in ids):
        return 1
    else:
        return 0
    
def get_ts( gz_filename ):
    return get_time( gz_filename[-13:-3] )   
    
def get_year( gz_filename ):
    return get_ts( gz_filename )[:4]    

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

def try_acct_file( test_date, jobid ):
    try:
        base = '/oasis/projects/nsf/sys200/stats/xsede_stats/comet_accounting/'
        end_date = str( test_date )
        jobid = str( jobid )
        
        test_file = base + end_date + '.txt'
        test_acct_info = [ line for line in open_txt( test_file ) if jobid in line ]
        
        return test_acct_info
    
    except:
        return []
    
def buffer_try_acct_file( a_search_tup, anon=True ):
    test_end_date = a_search_tup[2][:10]
    test_jobid = a_search_tup[3]
    test_ret = try_acct_file( test_end_date, test_jobid )
    anon_filter = ['User', 'Account', 'JobName']
    
    try:
        if (test_ret) and (not anon):
            acct_rules = 'JobID|User|Account|Start|End|Submit|Partition|Timelimit|JobName|State|NNodes|ReqCPUS|NodeList\n'.split('|')
            acct_data = test_ret[0].split('|')
            paired = { acct_rules[i] : acct_data[i] for i in range(len(acct_rules)) }
        
            return paired
        
        elif (test_ret) and anon:
            acct_rules = 'JobID|User|Account|Start|End|Submit|Partition|Timelimit|JobName|State|NNodes|ReqCPUS|NodeList\n'.split('|')
            acct_data = test_ret[0].split('|')
            paired = { acct_rules[i] : acct_data[i] for i in range(len(acct_rules)) if acct_rules[i] not in anon_filter }
        
            return paired            
    
    except:
        return test_ret

def fill_acct_info( partial_info, info_keys ):
    filled = { item:{} for item in info_keys }
    base = partial_info["Out"]
    
    for item in info_keys:
        base_item = base[ item ]
        filled[ item ] = buffer_try_acct_file( item )
        filled[ item ].update( base_item  )            
    
    return filled    

## !! RECURSION !!
## Used by deep_search_host()
## DESCRIPTION: Buffers contiguous 'bad' files
def try_open_x( aList, x ):
    try:
        unzip_txt( aList[x] )
        return ( aList[x], x )
    
    except:
        x += 1
        try_open_x( aList, x )

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

def collect_failed( acct_file, lim=False ):
    chunks = open_txt( acct_file )    
    test_ids = []

    if (lim) and (lim > 0):
        for chunk in chunks[1:]:
            if len(test_ids) >= lim:
                return test_ids
            else:
                state = chunk.split("|")[9]
                if state == "FAILED":
                    test_ids.add(chunk)
        
    else:
        for chunk in chunks[1:]:
            state = chunk.split("|")[9]
            if state == "FAILED":
                test_ids.append( chunk )
    
    return test_ids

def failed_search( collected_dict_list ):
    
    for i in range( collected_dict_list ):
        temp = collected_dict_list[i]
        nodelist = temp['NodeList']
        s = temp['Start']
        e = temp['End']
        jobid = temp['JobID']

def collect_headers( host_gzfile ):
    lines = unzip_txt( host_gzfile )
    ret_list = [ line for line in lines if 'comet' in line and ' ' in line ]
    return ret_list[1:]

def quick_save( obj, label=get_time() ):
    
    try:
        out_file = open( label, 'wb')
        pickle.dump( obj, out_file)
        
    except:
        "There was a problem pickling the object - Save manually."
        
def quick_load( pkl_file ):
    
    try:
        pkld_obj = open( pkl_file, 'rb' )
        obj = pickle.load( pkld_obj )
        
        return obj
        
    except:
        f"There was a problem unpickling an object from {pkl_file}."

####Formatting
def format_acct_file( str_date ):
    return conv + '/' + str_date + '.txt'

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

def format_header( line ):
    chunks = line.split(" ")
    
    try:
        return { "Timestamp": chunks[0], 
                     "Jobid": chunks[1],
                     "Host": chunks[2][:11] }
    except:
        return line
    
def format_data( line ):
    chunks = line.split(" ")
    
    stat = chunks[0]
    dev = chunks[1]
    data = chunks[2:-1]
    
    return { (stat,dev): data }

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

def labeled_data( data_list ):
    out_dict = {}
    
    for i in range( len( data_list )):
        line = data_list[i]
        label = line[:3]
        val = line[3]
        t = line[4]
        
        if label in out_dict:
            out_dict[label].append( ( t, val ) )
        else:
            out_dict[label] = [( t, val )]
    
    return out_dict

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

def acct_file_to_searchset( acct_chunks=False, partial_txt=False, full_txt=False ):
    out_list = []
    
    if acct_chunks:
        chunks = acct_chunks
    
    elif partial_txt:
        raw_chunks = open_txt( partial_txt )
        chunks = [ item for item in raw_chunks ]
    
    elif full_txt:
        chunks = open_txt( full_txt )[1:]
    
    else:
        return 1
        
    for i in range(len( chunks )):
        curr = chunks[i].split("|")
        jobid = curr[0]
        s = curr[3]
        e = curr[4]
        
        # actual reported host node(s)
        if "comet" in curr[-1]:
            
            if '\n' in curr[-1] or '\\n' in curr[-1]:
                check = curr[-1]
                checked = check.replace('\n','').replace('\\n','')
            else:
                checked = curr[-1]
                
            nodelist = format_nodelist( checked )
            obj_tup = ( nodelist, s, e, jobid )
            
            if len( obj_tup[0] ) > 1:
                exp_tups = separate_nodes( obj_tup )
                out_list += exp_tups
                    
            elif len( obj_tup[0] ) == 1:
                out_list.append( ( obj_tup[0][0], obj_tup[1], obj_tup[2], obj_tup[3] ) )
    
    return out_list

def buffer_schema( sch_set, w_id=False ):
    rules = sch_set[0]
    schema = rules.split(" ")
    data = sch_set[1:]
    
    if w_id:
        out_dict = { ea:{} for ea in schema }
        
        # TODO #
        
        #for line in sch_set[1:]:
        #    chunks = line.split(" ")[1:]
        #    
        #    for i in range(len(chunks)):
        #        out_dict[ schema[i] ] = chunks[i]
                
    else:
        out_dict = { name : [] for name in schema if 'FIXED_' in name } #name.partition(',')[0] to clean label
        
        for line in data:
            chunks = line.split(" ")[1:]
            
            for i in range( len(chunks)):
                if 'FIXED_' in schema[i]:
                    name = schema[i]                     #schema[i].partition(',')[0] to support clean labels
                    out_dict[ name ].append( float(chunks[i]) )     
    
    return out_dict

def find_recent_hosts( d0, dn, hosts_dict ):
    e = d0 + 'T00:00:00'
    s = dn + 'T23:59:59'
    out_dict = {}
    
    for host,host_list in hosts_dict.items():
        
        for host_file in host_list:
            t = get_time( host_file[-13:-3] )
             
            if t[:7] == s[:7]:
                
                if host in out_dict:
                    out_dict[host].append( host_file )
                    
                else:
                    out_dict[ host ] = [ host_file ]
                    
    return out_dict

def find_fixed( host_file ):    
    lines = unzip_txt( host_file )
    cpi_set = [ line for line in lines if 'intel_8pmc3' in line ]
    return cpi_set

def find_fixed_set( host_dict, lim=0 ):
    out_dict = { host_name:[] for host_name in host_dict.keys() }
    
    for host_name, file_list in host_dict.items():
        
        if lim == 0:
            for i in range(len( file_list )): 
                host_file = file_list[i]
                out_dict[ host_name ].append( find_fixed( host_file ) )
        else:
            cut = file_list[ : lim ]
            for i in range(len( cut )): 
                host_file = file_list[i]
                out_dict[ host_name ].append( find_fixed( host_file ) )
                
    return out_dict

####Data analysis

def timely_dict( info_dict ):
    timely_data = []
    
    for header,stat_dict in info_dict["Data"].items():
        chunks = header.split(" ")
        t = chunks[0]
        jobid = chunks[1]
        
        for key,data in stat_dict.items():
            stat = key[0]
            dev = key[1]
            
            for i in range(len(data)):
                metric = info_dict['Schemas'][stat][i]
                timely_data.append( (stat, dev, metric, data[i], t, jobid) )
    
    return timely_data

# Search a returned list of data for jobids
def filter_for_id( raw, t_id ):
    saved = []
    
    for i in range( len( raw )):
        line = raw[i]
        
        try:
            if int(line[5]) == int(t_id):
                saved.append( line )
        except:
            try:
                if str(line[5]) == str(t_id):
                    saved.append( line )
            except:
                pass
                
    return saved

####Data Munging

def info_dict( rules, info ):
    rules_list = rules.split("|")
    info_list = info.split("|")
    
    if len(rules_list) != len(info_list):
        return {}
    
    else:
        if '\n' in info_list[-1]:
            saved = info_list[:-1]
            extra = info_list[-1]
            cut = extra[:-1]
            
            info_list = saved.append( cut )
            
        return { rules_list[i]:info_list[i] for i in range(len(rules_list)) }

def job_to_info_dict( txt_file_list, target=False ):
    nodes_by_date = {}
    unsaved = []

    for date in txt_file_list:
        try:
            # skip alt files
            #check_stamp = int( date[-14] )
            #if (target) and (check_stamp == target):
            
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
            continue
            
    
    return nodes_by_date, unsaved    
    
def host_to_info_dict( zip_txt, jobid=0 ):
    l = zip_txt.find("comet")
    r = zip_txt.rfind("comet")
    
    if l != r:
        return zip_txt
    
    contents = unzip_txt( zip_txt )
    info_dict = { 
        "Schemas":{},
        "Specs":[],
        "Data":{},
        "Jobid(s)":[]
                }
    
    # Collect setting content
    for i in range(len( contents )):
        line = contents[ i ]
        
        # Spec line
        if line[0] == "$":
            info_dict["Specs"].append( format_spec( line ) )
        
        # Schema line    
        if line[0] == "!":
            info_dict["Schemas"].update( format_schema( line ) )
    
    # Collect variable content
    curr_header = ''
    
    for j in range( len(contents) ):
        line = contents[ j ]
        
        if "%" in line.split(" ")[0]:
            next
        
        # Header line
        elif check_header( line ):
            header_dict = format_header( line )
            curr_header = line[:-1]
                                        
            if "Timestamp" in header_dict:
                info_dict["Data"][ curr_header ] = {}
                    
                if ("Jobid" in header_dict) and (header_dict["Jobid"] not in info_dict["Jobid(s)"]):
                    info_dict["Jobid(s)"].append( header_dict["Jobid"] )
    
        # Data line
        else:
            try:
                curr_data = format_data( line )
                info_dict["Data"][curr_header].update( curr_data )
                    
            except:
                next
                
    data_dict = timely_dict( info_dict )
    info_dict["Data"] = data_dict
    
    if jobid != 0:
        try:
            temp = info_dict["Data"]
            info_dict["Data"] = filter_for_id( temp, jobid )
        except:
            pass
        
    return info_dict

def buffer_multi_hosts( source_dict, jobid=0 ):
    if source_dict["Source"]:
        src_list = source_dict["Source"]
        
        if jobid != 0:
            base_info = host_to_info_dict( src_list[0], jobid )
        else:
            base_info = host_to_info_dict( src_list[0] )
        
        if len( src_list ) == 1:
            base_info.update( source_dict )
            return base_info
    
        else:
            all_data = []
            all_ids = []
    
            for i in range(len( src_list )):
                src_file = src_list[i]
                
                if jobid != 0:
                    temp_info = host_to_info_dict( src_file, jobid )
                else:
                    temp_info = host_to_info_dict( src_file )
                
                data = temp_info["Data"]
                ids = temp_info["Jobid(s)"]
                
                for j in range(len( data )):
                    val = data[j]
                    all_data.append( val )
                    
                for j in range(len( ids )):
                    val = ids[j]
                    all_ids.append( val ) 

            out = {}
            out.update( source_dict )
            out["Schemas"] = base_info["Schemas"]
            out["Specs"] = base_info["Specs"]
            out["Data"] = all_data
            out["Jobid(s)"] = all_ids
            
            return out

def lookup_files( searchable_list ):
    found = []
    
    for m in range(len(searchable_list)):
        key = searchable_list[ m ]
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

def file_to_list( txt_file ):
    form_out = []
    
    with open( txt_file, "rt" ) as f:
        lines = f.read()
    f.close()
    
    list_items = [ item.replace('\n','').split(" ") for item in lines.split("\n\n") ]
    
    for i in range(len( list_items )):
        check = list_items[i]

        if len(check) >= 13:
            nodelist = format_nodelist( check[9].partition("=")[2] )
                
            s = check[7].partition("=")[2]
            e = check[8].partition("=")[2]
            jobid = check[0].partition("=")[2]
            
            this_search_tup = ( nodelist, s, e, jobid )
            
            if len(nodelist) == 1:
                form_out.append( ( nodelist[0], s, e, jobid ) )
                
            else:
                exp_tups = separate_nodes( this_search_tup )
                form_out += exp_tups
            
        
    return form_out

def deep_search_acct( file_list, search_list ):
    jobids = [ item[3] for item in search_list ]
    collected = {}
    
    #optional: predict filename using basename+namingrule
    
    for i in range(len(file_list)):
        try:
            possible = open_txt( file_list[i] )
            
            for j in range(len( jobids )):
                jobid = jobids[j]
                
                for chunk in possible:
                    if jobid in chunk:
                        collected[ search_list[j] ] = { "Data": chunk, "Source": file_list[i] }
        except:
            continue
    
    return collected

def deep_search_host( search_list, jobids=True ):
    
    for i in range(len( arc_data )):
        filename,x = try_open_x( host_file_list, i )
        file_year = get_year( filename )
        i=x
        
        headers = collect_headers( f )
        
        name_idx = f.find('comet-')
        host_name = f[ name_idx : name_idx + 11 ]
        
        for i in range(len( search_list )):
            target = search_list[ i ]
            target_host = target[0]
            target_id = target[3]
            
            if host_name == target_host:
                for header in headers:
                    
                    if target_id in header:
                        out[ target ]["Source"].append( f )         
                    else:
                        next
    
    return out

def deep_search_host_broken( search_list, jobids=False ):
    out = { item:{"Source":[] } for item in search_list }
    curr_year = get_time()[:4]
    
    for i in range(len( search_list )):
        target = search_list[ i ]#.split(" ")
        target_host = target[0]
        target_s = target[1]
        target_e = target[2]
        target_y = target_e[:4]
        
        if target_host not in sorted_arc_data:
            out[ target ]["Source"].append( "HostNotFound" )
        
        else:
            host_file_list = sorted_arc_data[ target_host ]
            
            if curr_year == target_y:
                host_file_list.reverse()
            
            for j in range( len( host_file_list ) ):
                filename,x = try_open_x( host_file_list, j )
                file_year = get_year( filename )
                j=x
            
                # screen for files already added to source collection
                if ( filename not in out[ target ]["Source"] ) and ( (file_year == target_s[:4] ) or ( file_year == target_e[:4] ) ):
                    headers = collect_headers( filename ) 
                    
                    # jobids included in search arguments
                    if jobids:
                
                        # if jobid is only id in file
                        if check_host_file( headers, target ) == 2 :
                            out[ target ]["Source"].append( filename )
                            
                            if ( "Single JobID" not in out[target] ):
                                out[ target ]["Single JobID"] = True
                                                                                        
                            ####just pull all data from file
                                
                        # if jobid is in file but with others
                        elif check_host_file( headers, target ) == 1:            
                            out[ target ]["Source"].append( filename )
                            out[ target ]["Single JobID"] = False
                                          
                            #### pull data but key:id, value:lines for id
                            
                        # jobid not found
                        else:
                            next
                    else:
                        next
            
            # jobids not included in search arguments
            # search by timestamps
            #else:
            #    for k in range(len( headers )):
            #        sample = headers[ k ].split(" ")
                    
                    # timestamp in header is between s/e timestamps in target
            #        if comp_window( target[1], target[2], sample[0] ) == 0:
                        
                        ####just pull all data from file
    
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
def search( mode=['s/e', 's', 'l','f'], from_list=False ):
    
    if mode == 's/e':
        t_0,t_n = input("Start, End:").replace(",", "").split(" ")
        start = get_stamp( t_0 )
        end = get_stamp( t_n )
        return start,end
    
    elif mode == 's':
        line = input("NL, Start, End:").replace(",", "").split(" ")
        line_tup = format_search_tup( line )
        return line_tup

    elif mode == 'l' and type( from_list ) is list:
        out_list = []
        dropped = []
            
        for obj in from_list:
            obj_tup = format_search_tup( obj )
            
            try:
                if len( obj_tup[0] ) > 1:
                    exp_tups = separate_nodes( obj_tup )
                    out_list += exp_tups
                    
                elif len( obj_tup[0] ) == 1:
                    out_list.append( obj_tup[0][0], obj_tup[1:] )
            
            except:
                try:
                    if ('comet' in obj[0]) and (len(obj[0]) == 11):
                        out_list.append( ( obj_tup[0][0], obj_tup[1], obj_tup[2], obj_tup[3] ) )
                except:
                    continue
                    
        #files,notFound = lookup_files( out_list )
        acct_info = deep_search_acct( acct_info_locs, out_list )
        host_info = deep_search_host( out_list, jobids=True )
        
        # prep output               
        out_dict = {}
        
        for i in range(len( out_list )):
            label = out_list[i]
            data_dict = { "Acct Info": acct_info[label]["Data"],
                          "Host Info": host_info[label]["Data"],
                          "Source Files": ( acct_info[label]["Source"], host_info[label]["Source"] )
                        }
            out_dict[ label ] = data_dict
                    
        return out_dict
    
    elif mode == 'f':
        search_list = group_from_text( input("Text file:") )
        return search_list
    
    else:
        return 1
    
def buffer_search_sample_n( sample_subset, subset_hosts ):
    # clean up unreturned search results
    purged = {}
    for key,val in subset_hosts.items():
        if (val["Source"]) and (val["Source"] != 'HostNotFound'):
            purged[key] = val
    
    return { 
        "Sample": sample_subset,
        "Raw": subset_hosts,
        "Out": purged
    }    

def search_sample_n( cut, n ):
    
    # read in search_out from saved dict in src_file
    cut = get_sample_hosts( search_set, n)
    sample_hosts = deep_search_host( cut, jobids=True )
    
    # format results
    out = buffer_search_sample_n( cut, sample_hosts )
    
    return out
   

def fill_host_info( search_sample ):
    if search_sample["Out"]:
        sample_dict = search_sample["Out"]
        keys = list(sample_dict.keys())
        sample_out = {}
        
        for i in range(len( keys )):
            key = keys[i]
            
            check = sample_dict[ key ]
            info_dict = buffer_multi_hosts( check, key[3] )
            sample_out[key] = info_dict
        
        return sample_out