import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ClusterTracker:
    def __init__(self, memory_window=15, backmatch_window=5,backmatch_treshold = 0.6):
        """
        Args:
            memory_window (int): How many past days to check for re-emerged clusters.
            backmatch_window (int): How many past days to check for robust cluster backward matching.
            backmatch_treshold (float): threshold for re-match, value range (0,1)
        """
        self.memory_window = memory_window
        self.backmatch_window = backmatch_window
        self.match_archive = {}  # global_id -> {"days": [..], "local_ids": {day: local_id}}
        self.next_global_id = 0
        self.day_to_global = {}  # {day: {local_id(day): global_id}
        self.all_dfs = {}
        self.backmatch_treshold = backmatch_treshold
        
    def track_single_day(self, date, df, output_path = None):
        """
        Args:
            daily_df (pd.DataFrame]: One dataframe per day with cols:
              ["cluster", "transition", "matching", "emerge"]
        Returns:
            updated dataframe with global cluster ID
        """
        
        
        if date not in self.day_to_global.keys():
            print(f"Initialize empty day to global mapping for {date}")
            self.day_to_global[date] = {}
        # emerged_clusters = df[df.emerge == True]
        # non_emerged_clusters = df[df.emerge != True]

        # assign gid to non emerged clusters (survived from the day before)
        df["gid"] = df["cluster"].apply(lambda x: self.day_to_global[date][x] if x in self.day_to_global[date].keys() else None)
        # for clusters assigned with gid, check
        df["gid"] = df.apply(lambda row: self.robust_backmatch(date, row["cluster"], row["gid"], row["ip"]) if pd.notna(row["gid"]) else row["gid"], axis=1)
        # assign gid to emerged clusters
        df["gid"] = df.apply(lambda row: 
                             row["gid"] if pd.notna(row["gid"]) 
                             else self.check_reemergence(date, row["cluster"], row["ip"])
                             , axis=1)

        next_date = self._get_day(1, date)
        self.day_to_global[next_date] = {}
        for _, row in df.iterrows():
            if row["transition"] == "survived":
                # directly link ID in the next day to gid of current date
                next_day_id = row["matching"][0]
                self.day_to_global[next_date][next_day_id] = row["gid"]

        self.all_dfs[date] = df

    def robust_backmatch(self, current_date, cluster, gid, ips):
        # need take care of gid if "false novelty" appears
        pastdays = [self._get_day(-i, current_date) for i in range(2,self.backmatch_window)]
        
        pastdays = [day for day in pastdays if day in self.day_to_global.keys()]#just to avoid no enough history for robust backmatch
        for past_day in pastdays:
            #perform conflict check only if it's not survived from that day
            if gid not in self.day_to_global[past_day].values():
                for ID in self.day_to_global[past_day]:
                    gid_past = self.day_to_global[past_day][ID]
                    # avoid more conflict if the one in the past survived until current date
                    if self.day_to_global[past_day][ID] not in self.day_to_global[current_date].values():
                        df_past = self.all_dfs[past_day]
                        ip_past = df_past.loc[df_past["cluster"] == ID, "ip"].iloc[0]
                        if self._is_match(ips, ip_past):
                            new_gid = gid_past
                            self.day_to_global[current_date][cluster] = new_gid
                            #  TO FIX, remove past values
                            # d = {k: ("y" if v == "x" else v) for k, v in d.items()}
                            # self.next_global_id = max(self.day_to_global[current_date].values())+1
                            
                            return new_gid
        return gid

    def check_reemergence(self, current_date, cluster_id,IPs):
        """Return global_id if matches past clusters, else assign a new gid."""
        pastdays = [self._get_day(-i, current_date) for i in range(2,self.memory_window)]
        pastdays = [day for day in pastdays if day in self.day_to_global.keys()]

        new_gid = None
        for past_day in pastdays:
            for ID in self.day_to_global[past_day]:
                gid_past = self.day_to_global[past_day][ID]
                if self.day_to_global[past_day][ID] not in self.day_to_global[current_date].values():
                    df_past = self.all_dfs[past_day]
                    ip_past = df_past.loc[df_past["cluster"] == ID, "ip"].iloc[0]
                    if self._is_match(ip_past, IPs):
                        new_gid = gid_past
        if new_gid == None:
            new_gid = self.new_global_id()
        new_gid = self.robust_backmatch(current_date, cluster_id, new_gid, IPs)
        self.day_to_global[current_date][cluster_id] = new_gid
        return new_gid
        
    def new_global_id(self):
        gid = self.next_global_id
        self.next_global_id += 1
        # self.archive[gid] = {"days": [], "local_ids": {}}
        return gid

    def update_archive(self, global_id, day, local_id):
        #To Do: delete old information?
        # build an archive?
        return None
    
    def _is_match(self, x, y):
        ol = len(set(x).intersection(set(y))) / len(x)
        if ol>=self.backmatch_treshold:
            return True
        else:
            return False

    def _get_day(self, i,current_date):
        current_date = datetime.strptime(current_date, '%Y%m%d')
        day = current_date+timedelta(days=i)
        day = day.strftime('%Y%m%d')
        return day

    def update_df_matching(self,savepath = None):
        #to do: Add a save option
        for date in self.all_dfs.keys():
            print(f"update {date}")
            next_day = self._get_day(1,date)
            if next_day in self.all_dfs.keys():
                self.all_dfs[date]["matching_gid"] = self.all_dfs[date]["matching"].apply(
                    lambda lst: [self.day_to_global[next_day][x] for x in lst])
                if savepath:
                    self.all_dfs[date].to_csv(f"{savepath}_{date}.csv", index=False)