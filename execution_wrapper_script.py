import os
import psutil
import sys
import timeit
import datetime
import shutil
from pathlib import Path

deploy_path = sys.argv[1]  # path
purpose = sys.argv[2]  # train or predict

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
logger=logger_obj.getLoggerInstance("Execution_Wrapper_Script")
job_st = JobStatus(deploy_path)


def train():
    log_date_fmt = "%Y-%m-%d %H:%M:%S.%f"
    step_name = 'TRAIN'
    start_time = datetime.datetime.now().strftime(log_date_fmt)
    job_st.update_job_start_time(step_name, start_time)

    logger.info("start training execution")
    train_exec = Train()
    job_status = train_exec.run()
    logger.info("end training execution")

    end_time = datetime.datetime.now().strftime(log_date_fmt)
    job_st.update_job_end_time_and_status(step_name, end_time, job_status)


def predict():
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


if __name__ == '__main__':
    start = timeit.default_timer()
    if purpose == 'train':
        train()
    elif purpose == 'predict':
        predict()
    else:
        print("Please provide second input parameter, train/predict.")
        logger.info("Please provide second input parameter, train/predict.")

    stop = timeit.default_timer()
    job_st.write_steps_dict_to_log()

    logger.info("Total Processing Time in seconds: " + str((stop - start)))

    process = psutil.Process(os.getpid())
    logger.info("Memory used in MB: " + str(((process.memory_info().rss)/1024)/1024))
    logger.info("Memory used in GB: " + str((process.memory_info()[0]/2.**30)))
