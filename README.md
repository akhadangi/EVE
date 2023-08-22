# EVE
This repository is official implementation of [**We Don't Need No Adam, All We Need Is EVE: On The Variance of Dual Learning Rate And Beyond**](https://arxiv.org/abs/2308.10740) using TensorFlow.

---

<font size = 4>EVE is equipped with two learning rates, where the 1st momentum is being obtained using short-term and long-term momenta, and 2nd momentum is the result of a residual learning paradigm. We have provided all the figures in interactive mode and you will be able to explore them on your ```DESKTOP``` browser by visiting [**EVE Optimiser Interactive Figures**](https://eve-optimiser.github.io). The users and the community are encouraged to download these files and visualise the performance of the Adam optimiser vs. EVE in more granular detail.



---
## **How to use?!**
<font size = 4>Assuming you have built a TensorFlow model named ```model```, you will be able to compile the model using EVE as follows:
```python
from eve import EVE
...
model.compile(optimizer = EVE(learning_rate_1 = 0.001, learning_rate_2 = 0.0004),
              loss = ..., metrics = ...)
history = model.fit(...)
...
```

---
### **side note for TPU usage**
<font size = 4>We have made the flower classification dataset available on GCS bucket. Follow the steps below to load the data into your Colab Notebook:
```python
import tensorflow as tf
# Detect hardware, return appropriate distribution strategy
# We aim to use TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

# Data access ...
GCS_DS_PATH = f"gs://flower-tpu"

# configuration ...
# We have used 224x224 pixels, options are one of either 192x192, 224x224, 331x331, 512x512
IMAGE_SIZE = [224, 224]
                       
img_size = IMAGE_SIZE[0]
EPOCHS = epochs
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

# Filaments ...
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition
```
