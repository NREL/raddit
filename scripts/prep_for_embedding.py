import pandas as pd
import json

def prep_string_for_llm(row):
    s = ''
    s += f'User: {row.user}\n\n'
    s += f'Account: {row.account}\n\n'
    s += f'Partition: {row.partition}\n\n'
    s += f'Job Type: {row.job_type}\n\n'
    s += f'Job Name: {row['name']}\n\n'
    s += f'QOS: {row.qos}\n\n'
    s += f'Submit Line: {row.submit_line}\n\n'
    s += f'Script: {row.script}\n\n'
    return s

df = pd.read_parquet('../data/historic_job_trace.parquet')

df['string_for_embedding'] = df.apply(prep_string_for_llm, axis=1)

for batch_number in tqdm(range(df.shape[0] // 4096 + 1)):
    start_index = batch_number * 4096
    end_index = (batch_number + 1) * 4096
    job_strings = df['string_for_embedding'][start_index:end_index].to_list()
    with open(f"../data/job_strings/job_strings_{batch_number}.json", "w") as f:
        json.dump(job_strings, f)