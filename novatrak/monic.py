import numpy as np
from copy import deepcopy
import pandas as pd

class Monic_with_noise:
    
    def __init__(self,tau = 0.6,tau_loose = 0.35):
        self.tau =tau
        self.tau_loose = tau_loose
        self.inactive = []
        self.disappeared = []
        self.splitted = []
        self.dict_splitted = {}
        self.matched = []
        self.absorbed = []
        self.survived = []
        self.emerged = []
        self.overlap_matrix = None
        self.overlap_matrix_reverse = None
        
    def _overlap(self, x, y):
        # overlap between 2 clusters, asymetric
        X = set(x)
        Y = set(y)
        inter = X.intersection(Y)
        # union = X.union(Y)
        return len(inter)/len(X), len(inter)/len(Y)
        
    def fit(self, t_0, t_1):    
        '''t_0 and t_1 are dataframes with column "cluster" '''
        clusters_t0 = t_0.cluster.drop_duplicates().sort_values().to_list()
        clusters_t1 = t_1.cluster.drop_duplicates().sort_values().to_list()
        # overlap matrix (len(c_0),y) = 
        overlap_matrix = np.zeros((len(clusters_t0), len(clusters_t1)))
        overlap_matrix_reverse = np.zeros((len(clusters_t1), len(clusters_t0)))

        # initialize overlap and backward woverlap matrix
        for i in range(len(clusters_t0)):
            for j in range(len(clusters_t1)):
                x = list(t_0.loc[t_0.cluster == clusters_t0[i]].index)
                y = list(t_1.loc[t_1.cluster == clusters_t1[j]].index)
                overlap = self._overlap(x, y)
                overlap_matrix[i,j] = overlap[0]
                overlap_matrix_reverse[j,i] = overlap[1]
        # overlape(d0,d1), [c0,c1]
        # row: cluster in day0, column: cluster in day1
        self.overlap_matrix = overlap_matrix
        # overlape(d1,d0), [c1,c0]
        # row: cluster in day1, column: cluster in day0
        self.overlap_matrix_reverse = overlap_matrix_reverse

        
        for c_0 in range(overlap_matrix.shape[0]-1):
            # noise is not considered, start from cluster 0, but its index for the matrix needs +1
            idx = c_0+1
            cluster_overlap = overlap_matrix[idx]
            
            # inactive, majority not active anymore i.e. A(x)< tau_loose
            
            active_overlap = np.sum(cluster_overlap)
            if active_overlap < self.tau_loose:
                self.inactive.append(c_0)
            
            # disappeared (strong match to noise)
            elif cluster_overlap[0] / active_overlap >= self.tau:
                self.disappeared.append(c_0)
                
            else:
                # start from now did not consider noise
                cluster_overlap = cluster_overlap[1:]

                # disappeared (no loose match)
                if max(cluster_overlap) / active_overlap < self.tau_loose:
                    self.disappeared.append(c_0)
                
                # split, exists loose match, no strong match
                elif max(cluster_overlap) / active_overlap < self.tau:
                    self.splitted.append(c_0)
                    # more details about split cluster (who are their "children")
                    self.dict_splitted[c_0] = list(np.nonzero(cluster_overlap / active_overlap > self.tau_loose)[0])
                    
                else:
                    # strong match exists [c0, c1_matched]
                    self.matched.append([c_0,np.argmax(cluster_overlap)])
                
        # if a cluster strongly matches a cluster in the next day, it can be survived or absorbed
        matched = np.array(self.matched)
        matched_Y, count_Y = np.unique(matched[:,1],return_counts=True)
        # identify clusters in d_1 which have more than 1 strongly match cluster from d_0
        Y_more_in_1 = matched_Y[count_Y > 1]
        # Y that has multiple X strongly match it are potentially emerged
        pot_emerged = deepcopy(Y_more_in_1)
        
        absorbed = []
        
        # survived clusters (1 to 1 strong match)
        survived_X_idx = np.array([i not in Y_more_in_1 for i in matched[:,1]])
        survived = matched[survived_X_idx]
        
        # select clusters c_0 that multiple clusters strongly match single Y
        multi_match_X_idx = np.array([i in Y_more_in_1 for i in matched[:,1]])
        more_in_1 = matched[multi_match_X_idx]

        for xy in more_in_1:
            idx_y = xy[1]+1
            idx_x = xy[0]+1
            # if X is a major component of Y, still survived
            if self.overlap_matrix_reverse[idx_y,idx_x]>=self.tau:
                survived = np.vstack([survived,xy])
                # if exists OL(Y,X)>= tau, then Y is not emerged
                pot_emerged = pot_emerged[pot_emerged != xy[1]]
            # otherwise abosrobed
            else:
                absorbed.append(xy)

        self.survived = survived
        self.absorbed = np.array(absorbed)


        ### identify emerged clusters Y
        # get all clusters in d_1, if not exist strong match, then emerged
        clusters_Y = np.arange(self.overlap_matrix.shape[1] - 1)
        
        unmatched_Y = clusters_Y[~np.isin(clusters_Y, matched_Y)]
        # combine all emerged (unmatched and multi-matched)
        self.emerged  = np.hstack((pot_emerged,unmatched_Y))
        
        
        
        # more info about emgerged clusters
        emerged_from_noise = []
        emerged_from_new = []
        for c1 in self.emerged:
            idx = c1 +1
            if overlap_matrix_reverse[idx,0] >= self.tau:
                emerged_from_noise.append(c1)
            elif sum(overlap_matrix_reverse[idx]) < self.tau_loose:
                emerged_from_new.append(c1)
        self.emerged_from_noise = emerged_from_noise
        self.emerged_from_new = emerged_from_new


    def get_transition_df(self):
        survived = getattr(self, "survived", [])
        inactive = getattr(self, "inactive", [])
        disappeared = getattr(self, "disappeared", [])
        absorbed = getattr(self, "absorbed", [])
        dict_splitted = getattr(self, "dict_splitted", {})
    
        cluster_ID = {
            "survived": survived[:, 0] if len(survived) > 0 else [],
            "inactive": inactive if len(inactive) > 0 else [],
            "disappeared": disappeared if len(disappeared) > 0 else [],
            "absorbed": absorbed[:, 0] if len(absorbed) > 0 else [],
            "splitted": list(dict_splitted.keys()) if dict_splitted else []
        }
    
        cluster_matched = {
            "survived": list(survived[:, 1].reshape(-1, 1)) if len(survived) > 0 else [],
            "inactive": [[] for _ in range(len(inactive))] if len(inactive) > 0 else [],
            "disappeared": [[] for _ in range(len(disappeared))] if len(disappeared) > 0 else [],
            "absorbed": absorbed[:, 1].reshape(-1, 1) if len(absorbed) > 0 else [],
            "splitted": list(dict_splitted.values()) if dict_splitted else []
        }

        cluster_trans_list = [(id_, trans) for trans, ids in cluster_ID.items() for id_ in ids]
        cluster_match_list = [id_m for _, ids in cluster_matched.items() for id_m in ids]
    
        df = pd.DataFrame(
            [(x[0], x[1], y) for x, y in zip(cluster_trans_list, cluster_match_list)],
            columns=["cluster", "transition", "matching"]
            )
        return df