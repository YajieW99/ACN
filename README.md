# ACN: Attentional  Compositional Networks for Long-Tailed Human Action Recognition

## The NEU-Interaction & Something-Else Annotations

- We collect the **NEU-Interaction dataset**, which currently contains **1365 video clips** in **16 different action classes**. There are **805 videos in the training set** and **560 videos in the testing set**. All the video clips are collected from a wide range of realistic scenes, and last from **2 to 4 seconds**. They are saved as mp4 files. The size of videos is 428*240 and frame rate is 15. We provide detailed annotations for the videos and action descriptions. For each video frame, we annotate the **human (hands)** and **action-related objects** with **bounding boxes** and their **identities**. For the action description, we annotate the **verbs**, **prepositions**, and the **adverbs** along with the action labels. 

- The annotations are provided in `/path/to/NEU-Interaction/annotations.json`.
  
- The annotation contains a dictionary mapping each video id, the name of the video file to the list of per-frame annotations.
  An example of per-frame annotation is shown below, 'video name/current frame' is given in the field 'name', 'labels' is a list of object's and hand's bounding boxes and names. 

```
   [
    {"name": "423/39.jpg", 
     "labels": 
              [{"box2d": {"x1": 283.0, "y1": 8.0, "x2": 419.0, "y2": 141.0}, 
                "standard_category": "hand"}, 
               {"box2d": {"x1": 168.0, "y1": 100.0, "x2": 244.0, "y2": 198.0}, 
                "standard_category": "0001"}, 
               {"box2d": {"x1": 182.0, "y1": 89.0, "x2": 227.0, "y2": 163.0}, 
                "standard_category": "0002"}, 
               {"box2d": {"x1": 227.0, "y1": 46.0, "x2": 418.0, "y2": 129.0}, 
                "standard_category": "0000"}]}, 
     {...},
     ...
     {...},
     ]
```

- We also provide new labels named 'verb-prep-labels' for both NEU-Interaction and Something-Something-v2 datasets. An example of is shown as follows. For each class id (e.g., '5'), the field 'classname' represents the corresponding action description (e.g., 'Putting something onto something'). The fields 'verb' and 'prep_adv' separately represent the verb and preposition/adverb (e.g., 'put' and 'onto') appearing in the action description. The fields 'verb_id' and 'prep_adv_id' represent the numeric id (e.g., 4 and 0) of the verb and preposition/adverb.


```
    {
      "5": {"classname": "Putting something onto something", 
            "verb": "put", 
            "verb_id": 4, 
            "prep_adv": "onto", 
            "prep_adv_id": 0},
    }
```
- All the action classes in our dataset and the verb and preposition/adverb annotations are shown as follows. 


|Class ID  |Class Description                         | Verb  |Verb ID| Prep/Adv    |Prep/Adv ID|
|----------|------------------------------------------|-------|-------|-------------|-----------|
|    0     |Pushing something onto something          | push  |   0   | onto        |   0       |
|    1     |Pulling something out of something        | pull  |   1   | out of      |   1       |
|    2     |Throwing something into something         | throw |   2   | into        |   2       |
|    3     |Moving something down                     | move  |   3   | down        |   3       |
|    4     |Pulling something from behind of something| pull  |   1   | behind      |   4       |
|    5     |Putting something onto something          | put   |   4   | onto        |   0       |
|    6     |Letting something roll up and down        | roll  |   5   | up and down |   5       |
|    7     |Moving something up                       | move  |   3   | up          |   6       |
|    8     |Letting something roll down               | roll  |   5   | down        |   3       |
|    9     |Moving something out of something         | move  |   3   | out of      |   1       |
|    10    |Throwing something onto something         | throw |   2   | onto        |   0       |
|    11    |Pulling something onto something          | pull  |   1   | onto        |   0       |
|    12    |Pouring something into something          | pour  |   6   | into        |   2       |
|    13    |Stuffing something into something         | stuff |   7   | into        |   2       |
|    14    |Putting something behind something        | put   |   4   | behind      |   4       |
|    15    |Pushing something and letting it drop down    | push  |   0   | down        |   3       |

- The verb-prep label for NEU-Interaction is `/path/to/NEU-Interaction/verb-prep-labels.json`

- The verb-prep label for Something-Something-v2 is `./Something-something-v2/something-something-v2-verb-prep-labels.json`

## Data Preparation

### NEU-Interaction

- Download the NEU-Interaction dataset we provide [here](https://drive.google.com/drive/folders/1xrKFBXP_2Rwh1gYMnf9AyXePnNW6EyxA?usp=sharing).

- Run `./NEU-Interaction_preprocess/frameNEU.py`, each video will be burst into frames in a separate folder in `/path/to/NEU-Interaction/frames/`. 
  We sample 16 frames per video and the chosen frames' index are saved as a txt file named after the video.
  The txt files are saved in a folder `/path/to/NEU-Interaction/list/`.

- To process the train data, run:

```
python frameNEU.py --root /path/to/NEU-Interaction/videos/ 
                    --json_file_input /path/to/NEU-Interaction/train.json 
                    --json_file_labels /path/to/NEU-Interaction/labels.json  
                    --save_path /path/to/NEU-Interaction/
```
- To process the test data, replace the input `train.json` with `test.json`.

### Something-Else

- Download the `something-something` database provided in the paper [The "something something" video database for learning and evaluating visual common sense](https://arxiv.org/pdf/1706.04261.pdf).

- Divide the database into the original task and the compositional task as described in the paper [Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9156858).

## Training & Testing

- To train the GNNs model for node feature fusion, run `./NodeFeature-GNNs/code/train.py`:

```
python train.py --logname experience_name --batch_size 72 
                --root_frames /path/to/Something-Else/20bn/
                --json_data_train /path/to/original/something-something-v2-train.json
                --json_data_val /path/to/original/something-something-v2-validation.json
                --json_file_labels /path/to/original/something-something-v2-labels.json
                --tracked_boxes /path/to/STHV2boundingbox/bounding_box_annotations.json
```

- To train the ACN model for the original Something-Else task, run `./ACN/train.py`:

```
python train.py --logname experience_name --batch_size 72 
                --root_frames /path/to/Something-Else/20bn/ 
                --json_data_train /path/to/original/something-something-v2-train.json
                --json_data_val /path/to/original/something-something-v2-validation.json
                --json_file_labels /path/to/original/something-something-v2-labels.json
                --json_new_labels /path/to/something-something-v2-verb-prep-labels.json
                --json_longtail /path/to/original/longtail30.json
                --tracked_boxes /path/to/STHV2boundingbox/bounding_box_annotations.json
```
- To evaluate the model, use the same script as training with a flag `--evaluate` and the checkpoint you have got `--resume /path/to/acn_model.pth.tar`.
- As for the compositional Something-Else task and the NEU-Interaction task, just change the corresponding input parameters.

