from datasets.cats_vs_dogs import get_catsvsdogs_dataset

class Args:
    def __init__(self):
        self.act_mode_8bit = True  # or False

args = Args()

train_ds, test_ds = get_catsvsdogs_dataset(("./data", args), load_train=True, load_test=True)

print("Number of training images:", len(train_ds))
for i in range(min(10, len(train_ds))):
    img, lbl = train_ds[i]
    print("Entry", i, ":", train_ds.data_list[i][0], "->", img.shape, "label:", lbl)
