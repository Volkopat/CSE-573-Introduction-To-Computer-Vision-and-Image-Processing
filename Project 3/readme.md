# Files in this folder:

1. <u>ComputeFBeta</u>

   It is used for compute f1 score between two json files.

2. <u>Json_Checker_Annotator</u>

   It is used to check if the generated the json file in correct format.

   It can also help visualize the json results if images are also provided.

3. <u>json_example</u>

   A demo to show how to create json file.

# How to run your code
## Part A
```python
# Face detection on validation data
python FaceDetector.py --input_path validation_folder/images --output ./results_val.json

# Validation
python3 ComputeFBeta/ComputeFBeta.py --preds results_val.json --groundtruth validation_folder/ground-truth.json

# Face detection on test data
python FaceDetector.py --input_path test_folder/images --output ./results.json
```

## Part B
```python
python FaceCluster.py --input_path faceCluster_5 --num_cluster 5
```

