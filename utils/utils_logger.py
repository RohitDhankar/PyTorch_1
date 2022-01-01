#https://docs.python.org/3/library/logging.html#levels
#https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
#pytorch_log_cnfg_dict -- >> 
#https://docs.python.org/3/howto/logging-cookbook.html#an-example-dictionary-based-configuration
#https://docs.python.org/3/library/logging.handlers.html#rotatingfilehandler



import os , logging 
import logging.config
from datetime import date
from datetime import datetime

LOGS_PATH = "./pytorch_logs_dir/"
LOG_LEVEL = "DEBUG"#"INFO" #https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
os.makedirs(LOGS_PATH, exist_ok=True)

def setup_logger(module_name=None, folder_name=None):
    dt_time_now = datetime.now()
    hour_now = dt_time_now.strftime("_%m_%d_%Y_%H")  
    #US Date Format == _%m_%d_%Y .... alternative date FMT == %d-%m-%Y
    
    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)
    hourly_log_path = os.path.join(LOGS_PATH, f'{hour_now}')
    if not os.path.exists(hourly_log_path):
        os.makedirs(hourly_log_path)
    
    if folder_name is None:
        module_log_file = os.path.join(hourly_log_path, f'{module_name}.log')
    else:
        if not os.path.exists(os.path.join(hourly_log_path, module_name)):
            os.makedirs(os.path.join(hourly_log_path, module_name))
        log_file_name = "pyTorch_log_"+hour_now+"00h_"
        module_log_file = os.path.join(hourly_log_path, module_name, f'{log_file_name}.log')

    pytorch_log_cnfg_dict = {
        'version': 1,
        "disable_existing_loggers": False,
        "formatters": {
            "pytorch_log_format": {
                "format": "%(asctime)s - %(levelname)s - Py_Module_Name: %(filename)s - Py_Method_Name: "
                          "%(funcName)s() - Code_Line: %(lineno)d Logged_Message:  %(message)s",
                "datefmt": "_%m_%d_%Y_%H:%M:%S"
            }
        },
        'handlers': {
            'common_handler': {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": LOG_LEVEL,
                "formatter": "pytorch_log_format",
                "filename": module_log_file,
                "when": "midnight",
                "interval": 1,
                "backupCount": 31,
                "encoding": "utf8"
            }
        
        },
        'loggers': {
            'general': {
                'level': LOG_LEVEL,
                'handlers': ['common_handler']
            }
        }
    }
    if module_name != 'general':
        module_handler = {
            'class': 'logging.FileHandler',
            'level': LOG_LEVEL,
            'formatter': "pytorch_log_format",
            'filename': module_log_file
        }
        module_logger = {
            'level': LOG_LEVEL,
            'handlers': [f'console_for_{module_name}','common_handler'] 
        }
        if f'console_for_{module_name}' not in pytorch_log_cnfg_dict['handlers'].keys():
            pytorch_log_cnfg_dict['handlers'][f'console_for_{module_name}'] = module_handler
        if module_name not in pytorch_log_cnfg_dict['loggers'].keys():
            pytorch_log_cnfg_dict['loggers'][module_name] = module_logger
    logging.config.dictConfig(pytorch_log_cnfg_dict)
    return logging.getLogger(module_name)


if __name__ == '__main__':
    pytorch_logger = setup_logger(module_name='logs_pytorch', folder_name=str('pytorch_logs_dir'))

