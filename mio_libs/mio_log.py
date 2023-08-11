import os
import os.path
import datetime
from enum import Enum
#
# Copyright 2022 by Vmio System JSC
# All rights reserved.
# System log implementation
#

class level_console(Enum):
    DEFAULT = 0
    INFO = 1
    DEBUG = 2
    WARNING = 3
    ERROR = 4
    EXCEPTION = 5
    CRITICAL = 6

class mio_log():
    def __init_from(self, path_dir):
        try:
            self.path_dir = path_dir
        except Exception as e:
            return str(e)
        return None

    def create(path_dir):
        log_obj = mio_log()
        if log_obj.__init_from(path_dir) is None:
            return log_obj
        return None

    def __name_file_today(self):
        current_date_and_time = datetime.datetime.now().strftime("%Y-%m-%d")
        return str(current_date_and_time)

    def __create_file_if_needed(self):
        path_file_log = os.path.join(self.path_dir, self.__name_file_today() + ".log")
        return path_file_log

    # Log some messages
    def debug(self, txt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        label = f'[{current_time}] [DEBUG     ] {txt}'
        print(label)

        path_file_log = self.__create_file_if_needed()

        append_log_endfile = open(path_file_log, "a")
        append_log_endfile.write(label)
        append_log_endfile.write("\n")
        append_log_endfile.close()

    def info(self, txt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        label = f'[{current_time}] [Info     ] {txt}'
        print(label)

        path_file_log = self.__create_file_if_needed()

        append_log_endfile = open(path_file_log, "a")
        append_log_endfile.write(label)
        append_log_endfile.write("\n")
        append_log_endfile.close()

    def warning(self, txt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        label = f'[{current_time}] [Warning     ] {txt}'
        print('\033[93m' + label + '\033[0m')

        path_file_log = self.__create_file_if_needed()

        append_log_endfile = open(path_file_log, "a")
        append_log_endfile.write(label)
        append_log_endfile.write("\n")
        append_log_endfile.close()

    def error(self, txt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        label =  f'[{current_time}] [Error     ] {txt}'
        print('\033[1;31m' + label + '\033[0m')

        path_file_log = self.__create_file_if_needed()

        append_log_endfile = open(path_file_log, "a")
        append_log_endfile.write(label)
        append_log_endfile.write("\n")
        append_log_endfile.close()

    def exception(self, ex:Exception, name_func: str):

        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        label = f'[{current_time}] [Exception     ] {name_func} at line {ex.__traceback__.tb_lineno} of {__file__}: {str(ex)}'
        print('\033[1;31m' +  label + '\033[0m')

        path_file_log = self.__create_file_if_needed()

        append_log_endfile = open(path_file_log, "a")
        append_log_endfile.write(label)
        append_log_endfile.write("\n")
        append_log_endfile.close()

    def critical(self, txt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        label = f'[{current_time}] [Critical     ] {txt}'
        print(label)

        path_file_log = self.__create_file_if_needed()

        append_log_endfile = open(path_file_log, "a")
        append_log_endfile.write(label)
        append_log_endfile.write("\n")
        append_log_endfile.close()

# Sample usage:
if __name__ == '__main__':
    log = mio_log.create("/Users/v-miodohien/Desktop/svn_mio_aicam/ServerLinux/source/logs")
    if log is not None:
        # 2. Log some messages
        log.debug("Debug message")
        log.info("Info message")
        log.warning("Warning message")
        log.error("Error message")
        log.exception(Exception('Test exceptions'))
        log.critical("Critical message")
    pass

