from os import environ
import uuid
import logging
import json

# file_name = environ('log_file_name')
file_name = 'app.log'


class ModelLog:

    def __init__(self):
        ModelLog.load_create()

    @staticmethod
    def request_uid(use_case):
        """
        Generate a unique unicode id for the object. The default implementation
        concatenates the class name, "_", and 12 random hex chars.
        """
        return str(use_case + "_" + uuid.uuid4().hex[12:])

    @staticmethod
    def load_create():
        logging.basicConfig(filename=('%s' % file_name), level=logging.INFO,
                            format='%(asctime)s ->%(message)s')
        logging.info('Started')

    @staticmethod
    def read_logs(uid):
        history = []
        with open(file_name) as reader:
            lines = reader.readlines()
            for line in lines:
                try:
                    val = json.loads(line.split('->')[-1])
                    if uid == val.get('name'):
                        history.append(val)
                except Exception as e:
                    print(e)
        return history

    @staticmethod
    def write_log(name, status, **kwargs):
        val = dict(name=name, status=status)
        val.update(kwargs)
        stringify = json.dumps(val)
        logging.info(stringify)


custom_log = ModelLog()
