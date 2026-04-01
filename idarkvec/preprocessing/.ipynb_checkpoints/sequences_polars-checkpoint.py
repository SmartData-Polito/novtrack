import polars as pl
class SequenceExtractor_pl():
    
    @staticmethod
    def _extract_by_ports(df, top_ports):
        df = df.with_columns(pp=pl.col("dst_port").cast(str)+'/'+pl.col("proto").cast(str))
        topN = df.group_by('pp').len().top_k(top_ports,by='len')['pp'].to_list()
        
        df = df.with_columns(
            pl.when(pl.col('pp').is_in(topN)).then(pl.col('pp')).otherwise(pl.lit("other"))
        )
        corpus = (
        df.group_by("pp")
        .agg(
            pl.col("src_ip").explode(),
        )
        .select("pp","src_ip")
        #.to_series().to_list()
    )
        # Extract IPs sequences by ports
        # sequences = df.sort_values('ts').groupby('pp')\
        #                                      .agg({'src_ip':list}).sort_index()
        
        return corpus