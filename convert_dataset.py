import numpy as np
import pandas as pd
import wfdb

# Dataset: https://physionet.org/content/mitdb/1.0.0/

files = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"
]

# https://archive.physionet.org/physiobank/annotations.shtml
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4897569/table/tab1/?report=objectonly
N = ("N", "L", "R", "e", "j")
S = ("A", "a", "J", "S")
V = ("V", "E")
F = ("F",)
Q = ("/", "f", "Q")
VALID_BEATS = N + S + V + F + Q

train = []
test = []
for file in files:
    record_path = f"mitbih/{file}"
    record = wfdb.rdrecord(record_path)
    
    if "MLII" not in record.sig_name: continue
    
    annotations = wfdb.rdann(record_path, "atr")

    symbols = []
    positions = []
    for symbol, position in zip(annotations.symbol, annotations.sample):
        if symbol in VALID_BEATS:
            symbols.append(symbol)
            positions.append(position)

    waveform = record.p_signal[:, record.sig_name.index("MLII")]
    rows = []
    for i in range(1, len(symbols) - 1):
        heartbeat = waveform[positions[i] - 90:positions[i] + 90]
        min_val, max_val = np.min(heartbeat), np.max(heartbeat)
        heartbeat = (heartbeat - min_val) / (max_val - min_val)
        
        if symbols[i] in N:
            attribute = 0.0
        elif symbols[i] in S:
            attribute = 1.0
        elif symbols[i] in V:
            attribute = 2.0
        elif symbols[i] in F:
            attribute = 3.0
        elif symbols[i] in Q:
            attribute = 4.0
        else:
            continue
        
        rows.append(np.append(heartbeat, [attribute]).astype(np.float32))
    
    for idx, row in enumerate(rows):
        if idx % 5 == 0:
            test.append(row)
        else:
            train.append(row)

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)
train_df.to_csv("mitbih_train.csv", header=None, index=False, float_format="%.9f")
test_df.to_csv("mitbih_test.csv", header=None, index=False, float_format="%.9f")
