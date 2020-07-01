import pandas as pd

tasks1 = pd.read_csv("../data/argue_corpus/qr_worker_answers_task1.csv")
tasks2 = pd.read_csv("../data/argue_corpus/qr_worker_answers_task2.csv")
annotated = pd.read_csv("../data/argue_corpus/qr_meta.csv")

posts = pd.read_csv("../data/argue_corpus/4forums.csv")

print(max(posts["disc_id"]))
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

posts.head()


tasks1.head()

posts[posts["disc_id"] == 10].head(50)


annotated = pd.merge(annotated, tasks1)
annotated = pd.merge(annotated, tasks2)

annotated.columns
annotated[annotated["agreement"] == max(annotated["agreement"])]
