# Create a function to generate predictions in a csv file
import pandas as pd
import numpy as np
def get_preds(model, test_data, filepath):
    '''
    This function take a model, a set of test data, and a file name.
    The model creates predictions based on the test data.
    Then the predictions are placed into a csv file acceptable for submission.
    '''
    # Make prediction and take the max value of each row
    preds = np.argmax(model.predict(test_data),axis=1) # set axis to 1 to analyze each row
    submission = pd.DataFrame({'ImageId':list(range(1,len(test_data)+1)),'Label':preds})
    submission.to_csv(filepath,index=False,header=True)