 
import pandas as pd
import numpy as np
import re
from hta.utils.utils import get_kernel_type, KernelType, merge_kernel_intervals
def get_training_kernels(trace_df,sym_table, search_item='cudaMalloc'):
    i_start= trace_df['ts'].min()
    i_end=trace_df['ts'].max()
    search_index = sym_table.index(search_item)
    searched_events= trace_df[trace_df['name']==search_index]
    #calculate ts percentage
    pctg=(searched_events['ts']- i_start)/(i_end-i_start)
    #print("the percentage of the searched evevnts")
    #print(pctg)
    training_kernels=trace_df[trace_df['ts']>searched_events['ts'].iloc[0] ]
    return training_kernels

def get_kernel_overlap(kernels):
    i_start= kernels['ts'].min()
    i_end=kernels['ts'].max()
    kernel_start=kernels['ts'][1:]
    kernel_end=kernels['ts']+kernels['dur']
    kernel_start=np.insert(kernel_start, -1, kernels['ts'].values[-1])
    idle_after=kernel_start- kernel_end
    return abs(idle_after[idle_after<0].sum())/(i_end-i_start)
    
def get_kernel_dur(kernel):
    i_start= kernel['ts'].min()
    i_end=kernel['end'].max()
    return i_end-i_start
def get_idle_dur(kernel):
    '''input kernel dataframe'''
    return kernel['idle_after'].sum()
def get_idle_precentage(kernel):
    i_start= kernel['ts'].min()
    i_end=kernel['ts'].max()
    #idle_inall=kernel['ts'].iloc[1:]- kernel['end'].iloc[0:-1].values
    idle_inall=kernel['idle_after'].sum()
    idle_pctg=idle_inall[idle_inall>0].sum()/(i_end-i_start)
    return idle_pctg

def get_idle_precentage_color(kernel):
    i_start= kernel['ts'].min()
    i_end=kernel['ts'].max()

def get_cpu_kernels(kernels):
    return kernels[kernels["stream"].eq(-1)].copy()
def get_gpu_kernels(kernels):
    return kernels[kernels["stream"].ne(-1)].copy()
def get_duration(kernels):
    return kernels['end'].max()-kernels['ts'].min()

def get_kernels_by_timepctg(kernels, start_pctg, end_pctg):
    ks=kernels['ts'].min()
    ke=kernels['end'].max()
    start_ts=ks+(ke-ks)*start_pctg
    end_ts=ks+(ke-ks)*end_pctg
    return kernels[(kernels['ts']>=start_ts) & (kernels['end']<=end_ts) ]

def set_event_end(kernels):
    endtime=kernels["ts"] + kernels["dur"]
    kernels['end']=endtime
def set_idle_before(kernels):
    kernel_end=kernels['end']
    kernel_end=kernel_end.iloc[0:-1].values
    kernel_end=np.insert(kernel_end, 0, kernels['ts'].values[0])
    idle_before= kernels['ts'].values- kernel_end
    kernels['idle_before']=idle_before
    
def set_idle_after(kernels):
    kernel_start=kernels['ts'][1:]
    kernel_end=kernels['ts']+kernels['dur']
    kernel_start=np.insert(kernel_start, -1, kernels['ts'].values[-1])
    idle_after=kernel_start- kernel_end
    kernels['idle_after']=idle_after

def set_idle_after_strict(kernels):
    
    kernel_end=kernels['ts']+kernels['dur']
    kernel_start_next= kernels['ts'][1:]
    
    #kernel_start=np.insert(kernel_start, -1, kernels['ts'].values[-1])
    idle_after=[]
    n=kernel_start_next.shape[0]
    for idx, endt in enumerate(kernel_end):
        newi=idx
        while newi<n and endt > kernel_start_next.iloc[newi]:
            newi=newi+1
        if newi==n:
            idle_after.append(0)
        else:
            idle_after.append(kernel_start_next.iloc[newi]-endt)
    
    #idle_after=kernel_start- kernel_end
    kernels['idle_after']=idle_after

def set_gpukernel_type(gpu_kernels, sym_table):
    gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
        lambda x: get_kernel_type(sym_table[x["name"]]), axis=1
    )

def get_comp_comm_mem_idle(gpu_kernels, sym_table):
    
    comp_kernels = gpu_kernels[
            gpu_kernels["kernel_type"].eq(KernelType.COMPUTATION.name)
        ].copy()

    comm_kernels = gpu_kernels[
            gpu_kernels["kernel_type"].eq(KernelType.COMMUNICATION.name)
        ].copy()
        
    mem_kernels = gpu_kernels[
            gpu_kernels["kernel_type"].eq(KernelType.MEMORY.name)
        ].copy()
    comp_dur=comp_kernels['dur'].sum()
    comm_dur=comm_kernels['dur'].sum()
    mem_dur=mem_kernels['dur'].sum()
    k_start=gpu_kernels['ts'].min()
    k_end=gpu_kernels['end'].max()
    span=k_end-k_start
    comp_pctg=comp_dur/span
    comm_pctg=comm_dur/span
    mem_pctg=mem_dur/span
    return comp_pctg, comm_pctg, mem_pctg, 1-mem_pctg-comm_pctg-comp_pctg


def get_timewindow_by_name(events_window, name):
    
    return events_window[events_window['name']==name]

def get_events_by_time_window(kernels, start_ts, end_ts):
    return kernels[(kernels['ts']>start_ts) & (kernels['ts']< end_ts)]



def get_events_window(matches_start_df, matches_end_df):
    


    # match start and end
    endlist=[]
    durlist=[]
    events_window=matches_start_df.copy()
    matches_end_new=matches_end_df.copy()
    for index, row in events_window.iterrows():
        #print(row[0], row[1], row[2])
        module_end=matches_end_new[(matches_end_new['name']==row[0]) & (matches_end_new['hash']==row[1])]
        endlist.append(module_end['end'].iloc[0])
        durlist.append(module_end['end'].iloc[0] - events_window['start'].iloc[index])

        '''
        for j, item in matches_end_new.iterrows():
            if row[0]==item['name'] and row[1]==item['hash']:
                endlist.append(item['end'])
                durlist.append(item['end'] - events_window['start'].iloc[index])
                #events_window['end'].iloc[index]=item[1]
                #events_window['dur'].iloc[index]=item[1]-events_window['start'].iloc[index]
                #matches_end_new.drop(index=j, inplace=True)
        '''

    events_window['end']=endlist
    events_window['dur']=durlist
    return events_window

def describe_kernels(kernels):
    '''
    can use this for all gpu kernels or for same event kernels
    describe the idle after and idle before
    '''
    #kernels['idle_before'].describe()
    kernels['idle_after'].describe()
    #know what events contribute to the idleness mostly
    #
    #gpu_events=

def count_name(sym_table):
    name_count = {}
    for name in sym_table:
        if name in name_count:
            name_count[name] += 1
        else:
            name_count[name] = 1
    return name_count

def select_events_with_count(name_count, n):
    selected_events = {name: count for name, count in name_count.items() if count == n}
    return selected_events



def position_in_trace(trace_df, ts_s, ts_e):
    start= trace_df['ts'].min()
    end= trace_df['end'].max()
    print("event in trace {}-{}".format((ts_s-start)/(end-start), (ts_e-start)/(end-start)))
    


def is_window_contain( win1, win2 ):
    '''
    input one line of timewindow dataframe
    '''
    return win1['start']<win2['start'] and win1['end']>win2['end']
def is_followed_by(win1, win2 ):
    return win1['end']<win2['start']

def is_window_contain_mul(win1, win2 ):
    '''
    win2 has multiple sequential layers
    '''
    win2_start=win2['start'].min()
    win2_end=win2['end'].max()
    return win1['start']< win2_start and win1['end']>win2_end
def is_followed_by_mul(win1, win2 ):
    win2_start=win2['start'].min()
    return win1['end']<win2_start

def is_window_contain_layerwise( win1, win2 ):
    '''
    both inputs have multiple lines
    '''
    return (win1['start'].to_numpy()<win2['start'].to_numpy()).all() and (win1['end'].to_numpy()>win2['end'].to_numpy()).all()

def overlapping_pctg(events_window):
    larger_start=pd.concat([pd.Series(events_window['start'].iloc[0:-1].values) , pd.Series(events_window['start'].iloc[1:].values)], axis=1).max(axis=1)
    smaller_end=pd.concat([pd.Series(events_window['end'].iloc[0:-1].values) , pd.Series(events_window['end'].iloc[1:].values)], axis=1).min(axis=1)
    overlap_dur=smaller_end- larger_start
    print(overlap_dur)
    
    overlap_pctg=overlap_dur/events_window['dur'].iloc[0:-1].values
    print(overlap_pctg)

def overlapping_pctg_strict(events_window):
    i_start= kernel['ts'].min()
    i_end=kernel['ts'].max()
    

    larger_start=pd.concat([pd.Series(events_window['start'].iloc[0:-1].values) , pd.Series(events_window['start'].iloc[1:].values)], axis=1).max(axis=1)
    smaller_end=pd.concat([pd.Series(events_window['end'].iloc[0:-1].values) , pd.Series(events_window['end'].iloc[1:].values)], axis=1).min(axis=1)
    overlap_dur=smaller_end- larger_start
    print(overlap_dur)
    
    overlap_pctg=overlap_dur/events_window['dur'].iloc[0:-1].values
    print(overlap_pctg)


def extract_values(log_contents):
    # Regular expressions to extract the required information
    ts_s_pattern = r'ts_s (\w+) at time: (\d+\.\d+)'
    ts_e_pattern = r'ts_e (\w+) at time: (\d+\.\d+) duration of (\w+): (\d+\.\d+)'

    ts_s_matches = re.findall(ts_s_pattern, log_contents)
    ts_e_matches = re.findall(ts_e_pattern, log_contents)

    # Create two lists to store the extracted values
    forward_data = []
    backward_data = []
    duration_data=[]
    # Process ts_s entries
    for idx, (group, timestamp) in enumerate(ts_s_matches):
        if group == 'forward':
            forward_data.append({'ts_s': int(float(timestamp)*1000000)})
        elif group == 'backward':
            backward_data.append({'ts_s': int(float(timestamp)*1000000)})

    # Process ts_e entries
    print(len(ts_s_matches),len(ts_e_matches))
    print(len(forward_data), len(backward_data))
    i_f=0
    i_b=0
    for idx, (group, timestamp,group_c,  duration) in enumerate(ts_e_matches):
        #print("group, direction, timestamp, duration:", group, direction, timestamp, duration)

        if group == 'forward':
            forward_data[i_f]['ts_e']=int(float(timestamp)*1000000)
            forward_data[i_f]['duration']=int(float(duration)*1000000)
            i_f=i_f+1
        elif group == 'backward':
            backward_data[i_b]['ts_e']=int(float(timestamp)*1000000)
            backward_data[i_b]['duration']=int(float(duration)*1000000)
            i_b=i_b+1
    #print(backward_data)
    # Create dataframes from the lists
    forward_df = pd.DataFrame(forward_data)
    backward_df = pd.DataFrame(backward_data)

    return forward_df, backward_df

def longest_common_subsequence(sequences):
    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        lcs_seq = []
        while lcs_length > 0 and m > 0 and n > 0:
            if X[m - 1] == Y[n - 1]:
                lcs_seq.append(X[m - 1])
                m -= 1
                n -= 1
                lcs_length -= 1
            elif dp[m - 1][n] > dp[m][n - 1]:
                m -= 1
            else:
                n -= 1

        return list(reversed(lcs_seq))

    if not sequences or any(not seq.size for seq in sequences):
        return []

    # Check that the sequences have similar lengths (within 1% of the average)
    avg_length = sum(len(seq) for seq in sequences) / len(sequences)
    for seq in sequences:
        if abs(len(seq) - avg_length) > 0.1 * avg_length:
            print("diff rate:",  abs(len(seq) - avg_length))
            raise ValueError("Sequences have significantly different lengths ")

    common_subseq = sequences[0]
    for i in range(1, len(sequences)):
        common_subseq = np.array(lcs(common_subseq, sequences[i]))
    return common_subseq

def layer_name2list(trace_df,events_window, sym_table, layer_name):
    layer_events=get_timewindow_by_name(events_window, layer_name)
    name_array=[]
    for index, row in layer_events.iterrows():
        events=get_events_by_time_window(trace_df, row['start'], row['end'])
        
        layer_gpu_kernels=get_gpu_kernels(events)
        
        name_list=layer_gpu_kernels['name'].values  #[ sym_table[x] for x in events['name']] 
        
        name_array.append( [ sym_table[x] for x in name_list])
        

    return name_array

def layer_name2pattern_lim1000(trace_df,events_window, sym_table, layer_name):
    layer_events=get_timewindow_by_name(events_window, layer_name)
    name_array=[]
    for index, row in layer_events.iterrows():
        events=get_events_by_time_window(trace_df, row['start'], row['end'])
        
        layer_gpu_kernels=get_gpu_kernels(events)
        
        name_list=layer_gpu_kernels['name'].values  #[ sym_table[x] for x in events['name']] 
        
        name_array.append(name_list)
        if index==1000:
            break
    if len(name_array)==1:
        return [ sym_table[x] for x in name_array[0]] 
    else:
        lcs=longest_common_subsequence(name_array)
        return [ sym_table[x] for x in lcs] 




def layer_name2pattern(trace_df,events_window, sym_table, layer_name):
    layer_events=get_timewindow_by_name(events_window, layer_name)
    name_array=[]
    for index, row in layer_events.iterrows():
        events=get_events_by_time_window(trace_df, row['start'], row['end'])
        
        layer_gpu_kernels=get_gpu_kernels(events)
        
        name_list=layer_gpu_kernels['name'].values  #[ sym_table[x] for x in events['name']] 
        
        name_array.append(name_list)
        
    if len(name_array)==1:
        return [ sym_table[x] for x in name_array[0]] 
    else:
        lcs=longest_common_subsequence(name_array)
        return [ sym_table[x] for x in lcs] 



def plot_eventdf(eventdf):
    '''
    input event time window dataframe
    '''
    hscatter_list=[]
    n=eventdf.shape[0]
    x_start=eventdf['start'].min()
    x_end=eventdf['end'].max()
    x_len=x_end-x_start
    for i in range(n):
        hscatter_list.append(np.array(np.zeros(x_len)))
        hscatter_list[i][eventdf['start'].iloc[i]: eventdf['end'].iloc[i]]=1
        
    fig, ax = plt.subplots()

    for i in range(n):
        # Get the binary data array for the current index 'i'
        #binary_data = hscatter_list[i]

        # Extract the indices where the value is 1
        #ones_indices = np.where(binary_data == 1)[0]

        # Plot vertical lines (or any marker of your choice) at the positions where value is 1
        ax.axhline(hscatter_list[i])#, color='b', linewidth=2

    # Set the y-axis ticks to represent the different arrays (0 to n-1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'Array {i}' for i in range(n)])

    # Set the x-axis range based on the length of the arrays
    ax.set_xlim(0, x_len)

    # Add labels and title
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Array Index')
    plt.title('Status of Each Point in n Arrays (0: Absent, 1: Present)')

    plt.show()
    




def get_func_name(name_string):
    '''
    only get first layer name
    '''
    p= r'(?:at::native::)?(?:\([^)]*\)::)?(\w+)[<(]'
    match = re.search(p, name_string)
    if match:
        return  match.group(1)
    else:
        return name_string
def get_func_name_stack(name_list):
    '''
    get first function name in a list from left to right
    '''
    p= r'(?:at::native::)?(?:\([^)]*\)::)?(\w+)[<(]'
    match = re.search(p, name_string)
    if match:
        return  match.group(1)
    else:
        return name_string

#import Levenshtein as lev

def namestr_similar(str1, str2):
    '''
    disable thisfunction
    '''
    print("namestr_similar is disabled for no version of Levenshtein")
    '''
    min_len = min(len(str1), len(str2))
    edit_distance = lev.distance(str1, str2)
    percentage_diff = edit_distance / min_len

    # Check if the percentage difference is smaller than 10%
    return percentage_diff < 0.10
    '''
    pass

def namestr_in(anchor, name):
    return anchor in name

def find_ts_by_anchor(trace_df, sym_table, anchor, comp_func=namestr_in):
    '''
    anchor is a string and we will find ts of all events in trace_df 
    whose name contains the anchor substring
    '''
    ts_list=[]
    for idx, row in trace_df.iterrows():
        if comp_func(anchor ,sym_table[int(row['name'])]):
            ts_list.append(row['ts'])
    return ts_list
    
def get_pctg_in_time(ts, start, end):
    return (ts-start)/(end-start)


def concatenate_strings(string_list):
    return ';'.join(string_list)

def find_elements_with_substring(strings, substring):
    '''
    name_list, string
    return the 
    '''
    result = []
    for string in strings:
        if substring in string:
            result.append(string)
    return result

def write_list_into_file(pattern_list, listname):
    '''
    write pattern list
    '''
    with open('./results/{}.txt'.format(listname),'w') as tfile:
        tfile.write('\n'.join(pattern_list))

def read_file_into_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]  # Removing trailing newline characters

        return lines
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

def collapse_subsequence_list(original_sequence, collapse_list):
    # find all collapse list in the original sequence
    event_sequence=original_sequence.copy()
    coll_len=len(collapse_list)
    union_name=concatenate_strings(collapse_list)
    seq_len=len(event_sequence)-coll_len
    i=0
    while i <seq_len :

        if(event_sequence[i]==collapse_list[0]):
            match=1
            for j in range(coll_len):
                if event_sequence[i+j]!=collapse_list[j]:
                    match=0
            if match==1:
                #replace these elements with a union element
                seq_len=seq_len-coll_len
                del event_sequence[i:i+coll_len]
                event_sequence.insert(i, union_name)
        i=i+1
    return event_sequence

def collapse_repeated_list(original_sequence, repeated_element):

    #count the repeated
    event_sequence=original_sequence.copy()
    seq_len=len(event_sequence)-1
    i=0
    while i < seq_len:
        
        if(event_sequence[i]== repeated_element):
            count=1
            j=i
            while j<seq_len and event_sequence[j]==repeated_element:
                count=count+1
                j=j+1

            if count>1:
                seq_len=seq_len-count
                del event_sequence[i:i+count]
                event_sequence.insert(i, "{"+repeated_element+" x"+str(count)+"}")
        i=i+1
    return event_sequence


def find_repeated_subsequence_continuous(sequence, subsequence_length, repeat_count):
    for start in range(len(sequence) - subsequence_length * repeat_count + 1):
        candidate = sequence[start : start + subsequence_length]
        if all(sequence[start + i * subsequence_length : start + (i + 1) * subsequence_length] == candidate for i in range(repeat_count)):
            return candidate
    return None

def find_repeated_subsequence(sequence, subsequence_length, repeat_count):
    for start in range(len(sequence) - subsequence_length * repeat_count + 1):
        candidate = sequence[start : start + subsequence_length]
        count = 1
        for i in range(start + subsequence_length, len(sequence) - subsequence_length + 1):
            if sequence[i : i + subsequence_length] == candidate:
                count += 1
                if count == repeat_count:
                    return candidate
    return None

def mask_numbers(input_string):
    result_string = re.sub(r'\d', '*', input_string)
    return result_string


def pattern_matching(event_name, pattern_item):
    '''
    see if event_name matches pattern_item
    '''
    return pattern_item in event_name



def find_pattern_in_trace_strict(trace_df, sym_table, pattern):
    '''
    the pattern is list of string
    return a list of df, which are subsets of trace_df
    
    '''
    results=[]
    plen=len(pattern)
    tlen=trace_df.shape[0]
    for i in range(tlen-plen):
        if pattern_matching(sym_table[trace_df['name'].iloc[i]], pattern[0]):
            mflag=True
            for j in range(0, plen):
                if not pattern_matching(sym_table[trace_df['name'].iloc[i+j]], pattern[j]):
                    mflag=False
                
            if mflag== True:
                results.append(trace_df.iloc[i:i+plen])
                
    return results


def draw_idle_forbackward(gpu_kernels, log_contents, figname="null"):
    forward_df, backward_df=extract_values(log_contents)
    
    forward_idle_pctg=[]
    forward_idle_time=[]
    forward_time=[]
    backward_idle_pctg=[]
    backward_idle_time=[]
    backward_time=[]

    for idx in range(30):
        forward_sample =  get_events_by_time_window(gpu_kernels, forward_df['ts_s'][idx],forward_df['ts_e'][idx] )
        backward_sample=get_events_by_time_window(gpu_kernels, backward_df['ts_s'][idx],backward_df['ts_e'][idx] )
        forward_idle_pctg.append(get_idle_precentage(forward_sample))
        forward_idle_time.append(get_idle_dur(forward_sample))
        forward_time.append(get_kernel_dur(forward_sample))
        #forward_overlap.append(get_kernel_overlap(forward_sample))

        backward_idle_pctg.append(get_idle_precentage(backward_sample))
        backward_idle_time.append(get_idle_dur(backward_sample))
        backward_time.append(get_kernel_dur(backward_sample))
        #backward_overlap.append(get_kernel_overlap(backward_sample))
        
    #plt.plot(forward_idle_pctg, label='Forward Idle Perce', linestyle='-', marker='o')
    plt.plot(forward_idle_time, label='Forward Idle Time (ms)', linestyle='-', marker='o')
    plt.plot(backward_idle_time, label='Backward Idle Time (ms)', linestyle='--', marker='^')
    plt.plot(forward_time, label='Forward Time (ms)', linestyle='-', marker='o')
    plt.plot(backward_time, label='Backward Time (ms)', linestyle='--', marker='^')
    #plt.plot(backward_overlap, label='Backward Overlap', linestyle='--', marker='^')

    # Adding labels and title
    plt.xlabel('Training step')
    plt.ylabel('Time in ms')
    #plt.title(' idleness and overlap of forward and backward kernels') 

    # Adding legend
    #plt.ylim(0, 1)
    plt.legend()

    # Display the plot
    if figname!='null':
        plt.savefig(figname)

    plt.show()