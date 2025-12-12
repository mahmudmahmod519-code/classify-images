from 'classifiy_image.py' import Classify_Image


data_set=[
    "/datasets/human",
    "/datasets/animals",
    "/datasets/planets",
]

model=Classify_Image(data_set[0],data_set[1],data_set[2])

model.fit_model()

model.test_accuorce(paths,y_test_numrical)