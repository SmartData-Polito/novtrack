import numpy as np
import polars as pl

def cos_numpy(vec,vec_1):
    return np.dot(vec, vec_1) / (np.linalg.norm(vec) * np.linalg.norm(vec_1))


def stat(vec):
    avg = np.mean(vec)
    std = np.std(vec)
    return avg, std


def volume(df, ip_list):
    # get avg #pkts and std per ip
    df_c = df.filter(pl.col("src_ip").is_in(ip_list))
    counts = df_c.group_by("src_ip").len().select("len").to_numpy().flatten()
    return stat(counts)


def inter_arr_time(df, ip_list):
    # filter relevant IPs
    df_c = df.filter(pl.col("src_ip").is_in(ip_list))
    # sort by timestamp
    df_c = df_c.sort("ts")
    # compute inter-arrival times per src_ip
    df_c = df_c.with_columns(pl.col("ts").diff().over("src_ip").alias("iat"))
    inter_arrival_means = df_c.group_by("src_ip").agg(pl.mean("iat").alias("mean_iat"))
    inter_arrival_means = inter_arrival_means.select("mean_iat").to_numpy().flatten()
    return stat(inter_arrival_means)


def num_dst_ips(df, ip_list):
    # filter relevant IPs
    df_c = df.filter(pl.col("src_ip").is_in(ip_list))
    # count unique destination IPs per source
    dst_counts =df_c.group_by("src_ip").agg(pl.col("dst_ip").n_unique().alias("unique_dst"))
    dst_counts =dst_counts.select("unique_dst").to_numpy().flatten()
    return stat(dst_counts)


def num_port_proto(df, ip_list):
    df_c = df.filter(pl.col("src_ip").is_in(ip_list))
    df_c = df_c.with_columns((
        pl.col("dst_port").cast(str) + "/" + pl.col("proto").cast(str)).alias("pp"))
    df_c = df_c.select('src_ip','pp')
    pp_counts =df_c.group_by("src_ip").agg(pl.col("pp").n_unique().alias("unique_pp"))
    pp_counts =pp_counts.select("unique_pp").to_numpy().flatten()
    return stat(pp_counts)

def 