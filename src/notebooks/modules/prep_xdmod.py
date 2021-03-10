######################################################################
# Formatted to work with data handled by ~/prep_IO.py
######################################################################
def cpi_calc( cpiset, cpi_hist ):
    core = cpiset[ 'FIXED_CTR1,E,W=48' ]
    inst = cpiset[ 'FIXED_CTR0,E,W=48' ]
    base = len(core)
    
    if cpi_hist:
        ctrs = { i : core[i] / inst[i] for i in range(len( core )) }
        return ctrs
    
    else:
        ctrs = [ core[i] / inst[i] for i in range(len( core )) ]
        return sum(ctrs) / base

# TODO
# make this work on ea in list and do cpi calculation
# still need to format to these keys and not prev
def fill_cpi( cpiset_list, cpi_hist=False ):
    out_dict = { i:[] for i in range(len( cpiset_list )) }
    
    for i in range(len( cpiset_list )):
        cpiset = cpiset_list[i]
        out_dict[ i ] = cpi_calc( cpiset, cpi_hist )
        
    return out_dict
######################################################################

## FROM RECREATE_XDMOD.IPYNB
def avg_blockbw(u):
    schema, _stats = u.get_type("block")
    blockbw = 0
    for hostname, stats in _stats.items():
        blockbw += stats[-1, schema["rd_sectors"].index] - stats[0, schema["rd_sectors"].index] + \
               stats[-1, schema["wr_sectors"].index] - stats[0, schema["wr_sectors"].index]
    return blockbw/(u.dt*u.nhosts*1024*1024)

def avg_cpi(u):
    schema, _stats = u.get_type("pmc")
    cycles = 0
    instrs = 0
    for hostname, stats in _stats.items():
        cycles += stats[-1, schema["CLOCKS_UNHALTED_CORE"].index] - \
            stats[0, schema["CLOCKS_UNHALTED_CORE"].index]
        instrs += stats[-1, schema["INSTRUCTIONS_RETIRED"].index] - \
            stats[0, schema["INSTRUCTIONS_RETIRED"].index] 
    return cycles/instrs

def avg_freq(u):
    schema, _stats = u.get_type("pmc")
    cycles = 0
    cycles_ref = 0
    for hostname, stats in _stats.items():
        cycles += stats[-1, schema["CLOCKS_UNHALTED_CORE"].index] - \
            stats[0, schema["CLOCKS_UNHALTED_CORE"].index]
        cycles_ref += stats[-1, schema["CLOCKS_UNHALTED_REF"].index] - \
                stats[0, schema["CLOCKS_UNHALTED_REF"].index] 
    return u.freq*cycles/cycles_ref

def avg_cpuusage(u):
    schema, _stats = u.get_type("cpu")    
    cpu = 0
    for hostname, stats in _stats.items():
        cpu += stats[-1, schema["user"].index] - stats[0, schema["user"].index]
    return cpu/(u.dt*u.nhosts*100)

def avg_ethbw(u):
    schema, _stats = u.get_type("net")
    bw = 0
    for hostname, stats in _stats.items():
        bw += stats[-1, schema["rx_bytes"].index] - stats[0, schema["rx_bytes"].index] + \
              stats[-1, schema["tx_bytes"].index] - stats[0, schema["tx_bytes"].index]
    return bw/(u.dt*u.nhosts*1024*1024)

def avg_fabricbw(u):
    avg = 0
    try:
        schema, _stats = u.get_type("ib_ext")              
        tb, rb = schema["port_xmit_data"].index, schema["port_rcv_data"].index
        conv2mb = 1024*1024
    except:
        schema, _stats = u.get_type("opa")  
        tb, rb = schema["PortXmitData"].index, schema["PortRcvData"].index
        conv2mb = 125000
    for hostname, stats in _stats.items():
        avg += stats[-1, tb] + stats[-1, rb] - \
               stats[0, tb] - stats[0, rb]
    return avg/(u.dt*u.nhosts*conv2mb)

def avg_flops(u):
    schema, _stats = u.get_type("pmc")
    vector_widths = {"SSE_D_ALL" : 1, "SIMD_D_256" : 2, 
                "FP_ARITH_INST_RETIRED_SCALAR_DOUBLE" : 1, 
                 "FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE" : 2, 
                 "FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE" : 4, 
                 "FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE" : 8, 
                 "SSE_DOUBLE_SCALAR" : 1, 
                 "SSE_DOUBLE_PACKED" : 2, 
                 "SIMD_DOUBLE_256" : 4}
    flops = 0
    for hostname, stats in _stats.items():
        for eventname in schema:
            if eventname in vector_widths:
                index = schema[eventname].index
                flops += (stats[-1, index] - stats[0, index])*vector_widths[eventname]
    return flops/(u.dt*u.nhosts*1e9)

def avg_l1loadhits(u):
    schema, _stats = u.get_type("pmc")
    load_names = ['LOAD_OPS_L1_HIT', 'MEM_UOPS_RETIRED_L1_HIT_LOADS']
    loads = 0
    for hostname, stats in _stats.items():
        for eventname in schema:
            if eventname in load_names:
                index = schema[eventname].index
                loads += stats[-1, index] - stats[0, index]
    return loads/(u.dt*u.nhosts)

def avg_l2loadhits(u):
    schema, _stats = u.get_type("pmc")
    load_names = ['LOAD_OPS_L2_HIT', 'MEM_UOPS_RETIRED_L2_HIT_LOADS']
    loads = 0
    for hostname, stats in _stats.items():
        for eventname in schema:
            if eventname in load_names:
                index = schema[eventname].index
                loads += stats[-1, index] - stats[0, index]
    return loads/(u.dt*u.nhosts)

def avg_llcloadhits(u):
    schema, _stats = u.get_type("pmc")
    load_names = ['LOAD_OPS_LLC_HIT', 'MEM_UOPS_RETIRED_LLC_HIT_LOADS']
    loads = 0
    for hostname, stats in _stats.items():
        for eventname in schema:
            if eventname in load_names:
                index = schema[eventname].index
                loads += stats[-1, index] - stats[0, index]
    return loads/(u.dt*u.nhosts)

def avg_lnetbw(u):
    schema, _stats = u.get_type("lnet")
    bw = 0
    for hostname, stats in _stats.items():
        bw += stats[-1, schema["rx_bytes"].index] + stats[-1, schema["tx_bytes"].index] \
              - stats[0, schema["rx_bytes"].index] - stats[0, schema["tx_bytes"].index]
    return bw/(1024*1024*u.dt*u.nhosts)

def avg_lnetmsgs(u):
    avg = 0
    schema, _stats = u.get_type("lnet")                  
    tx, rx = schema["tx_msgs"].index, schema["rx_msgs"].index

    for hostname, stats in _stats.items():
        avg += stats[-1, tx] + stats[-1, rx] - \
               stats[0, tx] - stats[0, rx]
    return avg/(u.dt*u.nhosts)

def avg_loads(u):
    schema, _stats = u.get_type("pmc")
    load_names = ['LOAD_OPS_ALL','MEM_UOPS_RETIRED_ALL_LOADS']
    loads = 0
    for hostname, stats in _stats.items():
        for eventname in schema:
            if eventname in load_names:
                index = schema[eventname].index
                loads += stats[-1, index] - stats[0, index]
    return loads/(u.dt*u.nhosts)

def avg_mbw(u):
    schema, _stats = u.get_type("imc")
    avg = 0
    for hostname, stats in _stats.items():
        avg += stats[-1, schema["CAS_READS"].index] + stats[-1, schema["CAS_WRITES"].index] \
             - stats[0, schema["CAS_READS"].index] - stats[0, schema["CAS_WRITES"].index]
    return 64.0*avg/(1024*1024*1024*u.dt*u.nhosts)

def avg_mcdrambw(u):      
    avg = 0
    schema, _stats = u.get_type("intel_knl_edc_eclk")
    for hostname, stats in _stats.items():
        avg += stats[-1, schema["RPQ_INSERTS"].index] + stats[-1, schema["WPQ_INSERTS"].index] \
             - stats[0, schema["RPQ_INSERTS"].index] - stats[0, schema["WPQ_INSERTS"].index]

    if not "flat" in u.job.acct["queue"].lower():
        schema, _stats = u.get_type("intel_knl_edc_uclk")
        for hostname, stats in _stats.items():
            avg -= stats[-1, schema["EDC_MISS_CLEAN"].index] - stats[0, schema["EDC_MISS_CLEAN"].index] + \
                stats[-1, schema["EDC_MISS_DIRTY"].index] - stats[0, schema["EDC_MISS_DIRTY"].index]

        schema, _stats = u.get_type("intel_knl_mc_dclk")
        for hostname, stats in _stats.items():
            avg -= stats[-1, schema["CAS_READS"].index] - stats[0, schema["CAS_READS"].index]

    return 64.0*avg/(1024*1024*1024*u.dt*u.nhosts)

def avg_mdcreqs(u):
    schema, _stats = u.get_type("mdc")
    idx = schema["reqs"].index
    avg = 0
    for hostname, stats in _stats.items():
        avg += stats[-1, idx] - stats[0, idx]
    return avg/(u.dt*u.nhosts)

def avg_mdcwait(u):
    schema, _stats = u.get_type("mdc")
    idx0, idx1 = schema["reqs"].index, schema["wait"].index
    avg0, avg1 = 0, 0 
    for hostname, stats in _stats.items():
        avg0 += stats[-1, idx0] - stats[0, idx0]
        avg1 += stats[-1, idx1] - stats[0, idx1]
    return avg1/avg0

def avg_openclose(u):
    schema, _stats = u.get_type("llite")
    idx0, idx1 = schema["open"].index, schema["close"].index
    avg = 0
    for hostname, stats in _stats.items():
        avg += stats[-1, idx0] - stats[0, idx0] + \
            stats[-1, idx1] - stats[0, idx1]
    return avg/(u.dt*u.nhosts)

def avg_oscreqs(u):
    schema, _stats = u.get_type("osc")
    idx = schema["reqs"].index
    avg = 0
    for hostname, stats in _stats.items():
        avg += stats[-1, idx] - stats[0, idx]
    return avg/(u.dt*u.nhosts)

def avg_oscwait(u):
    schema, _stats = u.get_type("osc")
    idx0, idx1 = schema["reqs"].index, schema["wait"].index
    avg0, avg1 = 0, 0 
    for hostname, stats in _stats.items():
        avg0 += stats[-1, idx0] - stats[0, idx0]
        avg1 += stats[-1, idx1] - stats[0, idx1]
    return avg1/avg0

def avg_packetsize(u):
    try:
        schema, _stats = u.get_type("ib_ext")              
        tx, rx = schema["port_xmit_pkts"].index, schema["port_rcv_pkts"].index
        tb, rb = schema["port_xmit_data"].index, schema["port_rcv_data"].index
        conv2mb = 1024*1024
    except:
        schema, _stats = u.get_type("opa")  
        tx, rx = schema["PortXmitPkts"].index, schema["PortRcvPkts"].index
        tb, rb = schema["PortXmitData"].index, schema["PortRcvData"].index
        conv2mb = 125000

    npacks = 0
    nbytes  = 0
    for hostname, stats in _stats.items():
        npacks += stats[-1, tx] + stats[-1, rx] - \
            stats[0, tx] - stats[0, rx]
        nbytes += stats[-1, tb] + stats[-1, rb] - \
            stats[0, tb] - stats[0, rb]
    return nbytes/(npacks*conv2mb)

def max_fabricbw(u):
    max_bw=0
    try:
        schema, _stats = u.get_type("ib_ext")              
        tx, rx = schema["port_xmit_data"].index, schema["port_rcv_data"].index
        conv2mb = 1024*1024
    except:
        schema, _stats = u.get_type("opa")  
        tx, rx = schema["PortXmitData"].index, schema["PortRcvData"].index
        conv2mb = 125000
    for hostname, stats in _stats.items():
        max_bw = max(max_bw, amax(diff(stats[:, tx] + stats[:, rx])/diff(u.t)))
    return max_bw/conv2mb

def max_lnetbw(u):
    max_bw=0.0
    schema, _stats = u.get_type("lnet")              
    tx, rx = schema["tx_bytes"].index, schema["rx_bytes"].index
    for hostname, stats in _stats.items():
        max_bw = max(max_bw, amax(diff(stats[:, tx] + stats[:, rx])/diff(u.t)))
    return max_bw/(1024*1024)

def max_mds(u):
    max_mds = 0
    schema, _stats = u.get_type("llite")  
    for hostname, stats in _stats.items():
        max_mds = max(max_mds, amax(diff(stats[:, schema["open"].index] + \
                                   stats[:, schema["close"].index] + \
                                   stats[:, schema["mmap"].index] + \
                                   stats[:, schema["fsync"].index] + \
                                   stats[:, schema["setattr"].index] + \
                                   stats[:, schema["truncate"].index] + \
                                   stats[:, schema["flock"].index] + \
                                   stats[:, schema["getattr"].index] + \
                                   stats[:, schema["statfs"].index] + \
                                   stats[:, schema["alloc_inode"].index] + \
                                   stats[:, schema["setxattr"].index] + \
                                   stats[:, schema["listxattr"].index] + \
                                   stats[:, schema["removexattr"].index] + \
                                   stats[:, schema["readdir"].index] + \
                                   stats[:, schema["create"].index] + \
                                   stats[:, schema["lookup"].index] + \
                                   stats[:, schema["link"].index] + \
                                   stats[:, schema["unlink"].index] + \
                                   stats[:, schema["symlink"].index] + \
                                   stats[:, schema["mkdir"].index] + \
                                   stats[:, schema["rmdir"].index] + \
                                   stats[:, schema["mknod"].index] + \
                                   stats[:, schema["rename"].index])/diff(u.t)))
    return max_mds

def max_packetrate(u):
    max_pr=0
    try:
        schema, _stats = u.get_type("ib_ext")              
        tx, rx = schema["port_xmit_pkts"].index, schema["port_rcv_pkts"].index
    except:
        schema, _stats = u.get_type("opa")  
        tx, rx = schema["PortXmitPkts"].index, schema["PortRcvPkts"].index

    for hostname, stats in _stats.items():
        max_pr = max(max_pr, amax(diff(stats[:, tx] + stats[:, rx])/diff(u.t)))
    return max_pr

# This will compute the maximum memory usage recorded
# by monitor.  It only samples at x mn intervals and
# may miss high water marks in between.   
def mem_hwm(u):
    # mem usage in GB
    max_memusage = 0.0 
    schema, _stats = u.get_type("mem")
    for hostname, stats in _stats.items():
        max_memusage = max(max_memusage, amax(stats[:, schema["MemUsed"].index] - \
                          stats[:, schema["Slab"].index] - \
                          stats[:, schema["FilePages"].index]))
    return max_memusage/(2.**30)

def node_imbalance(u):
    schema, _stats = u.get_type("cpu")
    max_usage = zeros(u.nt - 1)
    for hostname, stats in _stats.items():
        max_usage = maximum(max_usage, diff(stats[:, schema["user"].index])/diff(u.t))

    max_imbalance = []
    for hostname, stats in _stats.items():
        max_imbalance += [mean((max_usage - diff(stats[:, schema["user"].index])/diff(u.t))/max_usage)]    
    return amax([0. if isnan(x) else x for x in max_imbalance])

def time_imbalance(u):
    tmid=(u.t[:-1] + u.t[1:])/2.0
    dt = diff(u.t)
    schema, _stats = u.get_type("cpu")    
    vals = []
    for hostname, stats in _stats.items():
        #skip first and last two time slices
        for i in [x + 2 for x in range(len(u.t) - 4)]:
            r1=range(i)
            r2=[x + i for x in range(len(dt) - i)]
            rate = diff(stats[:, schema["user"].index])/diff(u.t)
            # integral before time slice 
            a = trapz(rate[r1], tmid[r1])/(tmid[i] - tmid[0])
            # integral after time slice
            b = trapz(rate[r2], tmid[r2])/(tmid[-1] - tmid[i])
            # ratio of integral after time over before time
            vals += [b/a]        
    if vals:
        return min(vals)
    else:
        return None

def avg_sf_evictrate(u):
    schema, _stats = u.get_type("cha")
    sf_evictions = 0
    llc_lookup = 0                  
    for hostname, stats in _stats.items():
        sf_evictions += stats[-1, schema["SF_EVICTIONS_MES"].index] - \
                  stats[0, schema["SF_EVICTIONS_MES"].index]
        llc_lookup   += stats[-1, schema["LLC_LOOKUP_DATA_READ_LOCAL"].index] - \
                  stats[0, schema["LLC_LOOKUP_DATA_READ_LOCAL"].index] 
    return sf_evictions/llc_lookup

def avg_page_hitrate(u):
    schema, _stats = u.get_type("imc")
    act = 0
    cas = 0                  
    for hostname, stats in _stats.items():
        act += stats[-1, schema["ACT_COUNT"].index] - \
             stats[0, schema["ACT_COUNT"].index]
        cas += stats[-1, schema["CAS_READS"].index] + stats[-1, schema["CAS_WRITES"].index] - \
             stats[0, schema["CAS_READS"].index] - stats[0, schema["CAS_WRITES"].index]
    return (cas - act) / cas

def max_sf_evictrate(u):
    schema, _stats = u.get_type("cha", aggregate = False)
    max_rate = 0
    for hostname, dev in _stats.items():    
        sf_evictions = {}
        llc_lookup = {}
        
        for devname, stats in dev.items():
            socket = devname.split('/')[0]
            sf_evictions.setdefault(socket, 0)
            sf_evictions[socket] += stats[-1, schema["SF_EVICTIONS_MES"].index] - \
                                    stats[0, schema["SF_EVICTIONS_MES"].index]
            llc_lookup.setdefault(socket, 0)
            llc_lookup[socket]   += stats[-1, schema["LLC_LOOKUP_DATA_READ_LOCAL"].index] - \
                                    stats[0, schema["LLC_LOOKUP_DATA_READ_LOCAL"].index]

    for socket in set([x.split('/')[0] for x in dev.keys()]):
        max_rate = max(sf_evictions[socket]/llc_lookup[socket], max_rate)
    return max_rate

def max_load15(u):
    max_load15 = 0.0 
    schema, _stats = u.get_type("ps")
    for hostname, stats in _stats.items():
        max_load15 = max(max_load15, amax(stats[:, schema["load_15"].index]))
    return max_load15/100