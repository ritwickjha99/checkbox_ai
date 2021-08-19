from django.shortcuts import render

import os
import psutil
import sys
import timeit
import datetime
import shutil
from pathlib import Path

deploy_path = r"C:/Users/lenovo/models/checkbox_ai/cb/" # path
# purpose = sys.argv[2]  # train or predict

sys.path.append(deploy_path)


from processing_module.scripts.common import utils
from processing_module.scripts.common.Logger import Logger
from processing_module.scripts.common import keywords
from processing_module.scripts.ensembler.train import Train
from processing_module.scripts.ensembler.predict import Predict
from processing_module.scripts.common.job_status import JobStatus

utility = utils.utils()
utility.readConfig(deploy_path)
Config = utility.getConfig()
logger_obj = Logger()
logger_config = logger_obj.createLogConfigObject(deploy_path)
logger = logger_obj.getLoggerInstance("Execution_Wrapper_Script")
job_st = JobStatus(deploy_path)

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# our home page view
def home(request):
    return render(request, 'index.html')


# custom method for generating predictions
def getPredictions():
    print("start Predict")
    log_date_fmt = "%Y-%m-%d %H:%M:%S.%f"
    step_name = 'PREDICT'
    start_time = datetime.datetime.now().strftime(log_date_fmt)
    job_st.update_job_start_time(step_name, start_time)

    logger.info("start predict execution")
    predict_exec = Predict()
    job_status = predict_exec.run()
    logger.info("end predict execution")

    end_time = datetime.datetime.now().strftime(log_date_fmt)
    job_st.update_job_end_time_and_status(step_name, end_time, job_status)


# our result page view

def result(request):
    
    folder = "C:/Users/lenovo/models/checkbox_ai/cb/media"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
  
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        getPredictions()
        return render(request, 'index.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'index.html')
