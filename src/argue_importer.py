from iacorpus import load_dataset

dataset = load_dataset('createdebate')
#print(dataset.dataset_metadata)
for discussion in dataset:
    print(discussion)
    for post in discussion[0]:
        print(post)
        print(type(post))
        exit()
