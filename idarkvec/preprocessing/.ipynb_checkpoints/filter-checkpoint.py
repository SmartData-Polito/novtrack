from subprocess import Popen, PIPE
import polars as pl

def count_daily_frequency(trace_file, min_pkts):
    print(f'[FILTER] Extracting the filter...')
    df = pl.read_csv(trace_file,row_index_offset=[0]).sort('ts')
    df = df.group_by('src_ip').len()
    filter_pl = df.filter(pl.col('len')>9).select("src_ip").to_series().to_list()
    print(f'         Dropped IPs sending less than {min_pkts} daily packets')
    print(f'         Retained IPs: {len(filter_pl)}')
    return filter_pl

# def count_daily_frequency(trace_file, min_pkts):
#     """_summary_

#     Parameters
#     ----------
#     trace_path : str
#         path of the raw traces file
#     min_pkts : int
#         minimum daily packets used in the filter

#     Returns
#     -------
#     list
#         list of daily IPs passing the filter
#     """
#     print(f'[FILTER] Extracting the filter...')
#     # Read the file
#     zcat_process = Popen(["zcat", trace_file], stdout=PIPE)
#     # Get the IP address field
#     cut_process = Popen(['cut', '-d', ',', '-f2'], 
#                         stdin=zcat_process.stdout, stdout=PIPE)
#     zcat_process.stdout.close()
#     # Sort IP addresses
#     sort_process = Popen('sort', 
#                          stdin=cut_process.stdout, stdout=PIPE)
#     cut_process.stdout.close()
#     # Count IP daily frequences
#     uniq_process = Popen(['uniq', '-c'], 
#                          stdin=sort_process.stdout, stdout=PIPE)
#     sort_process.stdout.close()
#     # Apply filter
#     awk_process = Popen(['awk', f'{{if($1>{min_pkts})print $2}}'], 
#                         stdin=uniq_process.stdout, stdout=PIPE)
#     uniq_process.stdout.close()
#     # Get the output
#     daily_frq = awk_process.communicate()[0].decode('utf-8').split('\n')
#     # Logs
#     print(f'         Dropped IPs sending less than {min_pkts} daily packets')
#     print(f'         Retained IPs: {len(daily_frq)}')
    
#     return daily_frq