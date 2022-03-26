from datasets import list_datasets
from datasets import load_dataset
from hhpy import display_df

_datasets = list_datasets()
print(f"there are {len(_datasets)} datasets")
print(f"first 10 datasets : {_datasets[:10]}")

_emotion_dt = list_datasets(with_details=True)[_datasets.index("emotion")]
print(f"Description : {_emotion_dt.description}")
print(f"Citation : ", "\n".join(_emotion_dt.citation.split("\n")[:8]))

emotions = load_dataset("emotion")
print(emotions)

emotions_train = emotions["train"]
print(f"datatype of emotions train set : {type(emotions_train)}")
print(f"length of emotions train set : {len(emotions_train)}")
print(f"first index of emotions train set : {emotions_train[0]}")
print(f"column names of emotions train dataset : {emotions_train.column_names}")
print(f"features of emotions train dataset : {emotions_train.features}")
print(f"first five rows of emotions train dataset : {emotions_train[:5]}")

# to convert train dataset to pandas dataframe
emotions.set_format("pandas")
emotions_df = emotions['train'][:]
display_df(emotions_df.head(), index=None)


# to convert integer labels to string
def label_int2str(label_int, split="train"):
    return emotions[split].features["label"].int2str(label_int)


emotions_df["label_name"] = emotions_df["label"].apply(label_int2str)
display_df(emotions_df.head(), index=None)

