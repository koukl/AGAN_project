### Packages Needed
Need PyTorch, Numpy

### dataset
Let's say your dataset is in a folder called data. To run the code you need create another folder, lets call it dataRoot, and place data into dataRoot. So the hierachy is ./dataRoot/data/image.png. This is done so the dataloader can correctly load the dataset.

### Running
A sample run will be as follows. You can open up main.py to check out the options you can set. Ask me if you're unsure.
> python main.py --outDir='out' --percent=0.5 --dataRoot='./dataRoot' --imageSize=128

outDir refers to the directory you store your weights/output image
percent refers to the keep history percent
