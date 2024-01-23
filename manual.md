# PROJECT STRUCTURE

AVProject/
├── PAR
│   ├── binary_classifier.py
│   ├── cbam.py
│   ├── convnext_extractor.py
│   └── par_utils.py
├── TOOLS
│   ├── annoucer.py
│   ├── person.py
│   ├── result_writer.py
│   └── roi.py
├── data
│   ├── par_datasets


# WHAT DO WE HAVE AND HOW IT WORKS

## ROI
Class RoiReader creates 2 Roi objects from a file given by professor.
Class Roi have only one method: include. It works that we give it a bounding box and it tells us if at least one point os bbox include to roi

## PERSON
Peron is a object which we create for each uniuqe person which we detect. It stores a attribiutes value in public variables, because all of the public variables are write to answear. Object also performs time in rois counting. My idea is that in each iteration of detector we give a information to object if the person is in roi and if object detect change of presence in roi it do the actions.

## RESULT WRITER
This object simply takes a iterable of Person objects, takes from each public variables and save it to file

## ANNOUCER
Object which has a method: annouce. This method should takes a person and place where this person apears and annnouce (no matter how).

## PAR MODEL
Each of PAR model takes a features from the same features extractor and should give a binary answear(hat,bag,gender) or multiclass answear(upper color,lower color). After training all the models will be merged into one model with shared features extractor but they are trained seperetaly with attention module.

BinaryClassiefier takes the features from ConvenextExtractor and gives answear True/False

ConvenextExtractor is the features extractor

CBAM is spatial and channel attention module

PAR is a wrapper for trained models and extractor to make shared feature model

ImageDataset is a PyTorch dataset which takes annotation file, directory with images, class which we are training (and choose only this images with anotated this class),and a transforms


## ENGINE
For now engine in each iteration detect and track people and for each NEW person is creating new object Person and classifies attribuites (so it computes only one the attributes and later they are always the same). In each iteration for each person detector talks if the person is in rois (it isn't the best idea) 


