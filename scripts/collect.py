import json
import pandas as pd
import re

def load_jsonl_or_json(path):
    with open(path, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            return json.loads(content)
        else:
            return [json.loads(line) for line in content.splitlines() if line.strip()]

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    return text

def main():
    injections = load_jsonl_or_json("data/raw/injections.json")
    benign = load_jsonl_or_json("data/raw/benign.json")  # adjust filename to match yours

    inj_df = pd.DataFrame(injections)
    inj_df['label'] = 1  # 1 = injection
    ben_df = pd.DataFrame(benign)
    ben_df['label'] = 0  # 0 = benign

    df = pd.concat([inj_df, ben_df], ignore_index=True)

    # normalize column name — adjust 'text' if your source files use a different key
    if 'text' not in df.columns:
        raise ValueError(f"Expected a 'text' column, found: {df.columns.tolist()}")

    df['text'] = df['text'].apply(clean_text)
    before = len(df)
    df = df.drop_duplicates(subset='text')
    df = df[df['text'].str.len() > 3]  # drop empty/junk rows
    after = len(df)
    print(f"Deduped: {before} -> {after} rows")

    print("\nLabel distribution:")
    print(df['label'].value_counts())

    df.to_csv("data/processed/dataset_v1.csv", index=False)
    print(f"\nSaved to data/processed/dataset_v1.csv")

if __name__ == "__main__":
    main()