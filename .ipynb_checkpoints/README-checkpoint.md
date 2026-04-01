# NovTrack
This repository contains the implementation of paper "NovTrack: Dynamic Cluster Analysis for Tracking Novelty in Network Telescopes"

## Install

1. Clone this repository
2. Create and activate a virtual environment (recommended):

```bash
cd novtrack
python -m venv ./.venv
source ./.venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## The pipeline

 - `data/parsepcap.ipynb` provides function to parse `.pcap` file with `dpkt`
 - You can follow the steps to train sender embeddings (`generate_embeddings.ipynb`), cluster the senders (`run_clustering.ipynb`) and track cluster changes ((`run_tracking.ipynb`))


## Dataset

Our dataset is available upon request, please contact the authors to get access

