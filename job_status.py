import os
import pandas as pd


class JobStatus(object):
    def __init__(self, deploy_path):
        self.deploy_path = deploy_path
        self.job_log_file = '_python.log'
        self.step_num = 'STEP_NUM'
        self.job_step_status = 'JOB_STEP_STATUS'
        self.start_dtm = 'START_DTM'
        self.end_dtm = 'END_DTM'

        self.steps_list = ['TRAIN', 'PREDICT']
        self.dict_df = pd.DataFrame()
        self.dict_df_cols = ['STEP_NUM', 'JOB_STEP', 'JOB_STEP_STATUS', 'START_DTM', 'END_DTM']

        self.steps_dict = {}
        self.__create_job_progress_dict()
        self.job_log_path = self.__get_job_log_file_path()

    def __create_job_progress_dict(self):

        for step_name in self.steps_list:
            step_details_dict = dict()
            step_details_dict[self.job_step_status] = 'SUCCESS'
            step_details_dict[self.start_dtm] = None
            step_details_dict[self.end_dtm] = None
            if step_name == 'TRAIN':
                step_details_dict[self.step_num] = 41
            elif step_name == 'PREDICT':
                step_details_dict[self.step_num] = 42

            self.steps_dict[step_name] = step_details_dict

    def update_job_status(self, step_name=None, is_success=None):
        if step_name is not None:
            if is_success is not None:
                self.steps_dict[step_name][self.job_step_status] = ('SUCCESS' if is_success == True else 'FAILED')

    def update_job_start_time(self, step_name=None, start_time=None):
        if step_name is not None:
            if start_time is not None:
                self.steps_dict[step_name][self.start_dtm] = start_time

    def update_job_end_time_and_status(self, step_name=None, end_time=None, is_success=None):
        if step_name is not None:
            if end_time is not None:
                self.steps_dict[step_name][self.end_dtm] = end_time
            if is_success is not None:
                self.steps_dict[step_name][self.job_step_status] = ('SUCCESS' if is_success else 'FAILED')

    def __get_job_log_file_path(self):
        job_log_relative_path = self.job_log_file
        job_log_file_path = os.path.join(self.deploy_path, job_log_relative_path)
        return job_log_file_path

    def write_steps_dict_to_log(self):
        self.dict_df = pd.DataFrame.from_dict(self.steps_dict, orient='index')
        self.dict_df['JOB_STEP'] = self.dict_df.index
        self.dict_df = self.dict_df[self.dict_df_cols]
        self.dict_df.sort_values(by=['STEP_NUM'], inplace=True)
        self.dict_df.to_csv(self.job_log_path, index=False)

