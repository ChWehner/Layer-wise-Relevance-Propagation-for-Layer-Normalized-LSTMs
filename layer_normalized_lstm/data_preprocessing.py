
import pandas as pd
import utility
import time
import glob

#loading dataset
for filename in glob.glob("split/*.json"):
    try:
    	df = pd.read_json(filename, encoding='utf-8')

    	start_time = time.time()

    	print("building new dataset")
    	df = utility.buildDf(df)

    	now_time= time.time()
    	print("--- %s seconds ---" % (now_time - start_time))
    	start_time = now_time

    	print("cleaning dataset")

    	df = utility.cleanX(df)
    	now_time= time.time()
    	print("--- %s seconds ---" % (now_time - start_time))

    	start_time = now_time
    	df = utility.trimYOutlier(df)
    	now_time= time.time()
    	print("--- %s seconds ---" % (now_time - start_time))
    	start_time = now_time

    	print("preprocess dataset and write to files")
    	df = utility.preproc(df)
    	
    except:
        print("could not read json")



