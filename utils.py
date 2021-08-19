import os
import json
from processing_module.scripts.common import keywords
import warnings
warnings.filterwarnings("ignore")


class utils:
    Config = None
    ConfigObj = None

    def getCompletepath(self, path):
        rootPath = self.Config['cb_ml_common']['installation_dir']
        completePath = str(rootPath) + path
        return completePath

    def getConfig(self):
        return utils.ConfigObj

    def get(self, step, param):
        return str(utils.Config[step][param])

    def readConfig(self, deploy_path):
        utils.ConfigObj=utils()
        with open(os.path.join(deploy_path, keywords.ML_CONFIG_FILE)) as json_file:
            json_file=json_file.read()
            data = json.loads(json_file)
            json_keys = list(data.keys())
            json_keys = [key for key in json_keys if key != 'global_params']
            cb_ml_common_dict = data['global_params']
            for component in json_keys:
                component_dict = data[component]
                for key, val in component_dict.items():
                    for cb_key in list(cb_ml_common_dict.keys()):
                        if cb_key in val:
                            replace_val = cb_ml_common_dict[cb_key]
                            val = val.replace(cb_key, replace_val)
                        data[component][key] = val
                        utils.Config=data








