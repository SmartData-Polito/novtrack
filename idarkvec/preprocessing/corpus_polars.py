import numpy as np
import polars as pl
from .filter import count_daily_frequency
from .sequences_polars import SequenceExtractor_pl

class BaseCorpus_pl:
    """_summary_

    Parameters
    ----------
    trace_path : _type_
        _description_
    day : _type_
        _description_
    min_freq : _type_
        _description_
    
    Attributes
    ----------
    trace_path : _type_
        _description_
    day : _type_
        _description_
    min_freq : _type_
        _description_
    """

    def __init__(self, trace_path, day, min_freq):
        self.min_freq = min_freq
        self.trace_path = trace_path
        self.day = day
        self.trace_file = f'{trace_path}_{day}.csv.gz'
        # self.trace_file = f'{trace_path}_{day}.csv'

    def _filter_trace(self):
        _filter = count_daily_frequency(self.trace_file, self.min_freq)
        # Load raw trace logs
        df = pl.read_csv(self.trace_file,row_index_offset=[0]).sort('ts')
        # Apply filter
        df = df.filter(pl.col("src_ip").is_in(_filter))

        return df

    def _rearrange_sequences(self, sequences):
        sequences = [self._drop_duplicates(x) for x in sequences.iter_rows()]
        # Manage final corpus
        sequences.sort(key=lambda x:x[0])
        corpus = [x[1] for x in sequences]

        return corpus

    def _drop_duplicates(self, x):
        order, service = x
        _prev = np.array(service)
        _next = np.roll(_prev, -1)
        _next[-1] = 'NULL'
        document = _prev[_prev!=_next]

        return (order, list(document))
        


class CorpusExtractor_pl(BaseCorpus_pl):
    """
    Methods
    -------
    from_darknet(top_ports)
        _description_

    """
    __doc__ = BaseCorpus_pl.__doc__ + __doc__

    def from_darknet(self, top_ports, verbose=False):
        """_summary_

        Parameters
        ----------
        top_ports : _type_
            _description_
        verbose : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        trace = self._filter_trace()
        print(f'[CORPUS] Extracting the corpus...')
        ip_sequences = SequenceExtractor_pl._extract_by_ports(trace, top_ports)
        corpus = self._rearrange_sequences(ip_sequences)

        if verbose:
            raw_words = np.hstack(corpus)
            effective_words = np.unique(raw_words)
            print(f'         Corpus extracted')
            print(f'         {raw_words.shape[0]} raw words')
            print(f'         {effective_words.shape[0]} effective words')
        
        return corpus

    def from_honeypot(self):
        """ Not implemented yet
        """
        pass