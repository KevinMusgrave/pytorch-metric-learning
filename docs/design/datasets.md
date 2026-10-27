# Dataset adapters

Dataset adapters translate published metadata formats into image paths and
labels. The train/test split is a property of the dataset: CUB separates the
first and second hundred classes, Stanford Online Products uses its train/test
metadata files, and the other adapters retain their documented splits.

Local fixture tests cover metadata parsing, split selection, transforms, and
image loading using temporary files. Full public-dataset downloads remain an
explicit opt-in through `TEST_DATASETS=true`; a skipped download is not evidence
that the provider URL or full dataset was verified.

Image decoding is a native-code trust boundary. File extensions do not constrain
Pillow's decoder selection. Dependency review must consider the actual
`Image.open` and transform paths, including caller-supplied dataset directories,
instead of assuming that every image is trusted because a common dataset
normally contains JPEG files.
