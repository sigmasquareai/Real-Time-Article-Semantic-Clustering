from configparser import ConfigParser

_FILEPATH = 'database.ini'
_CONFIG_SECTIONS = ['postgresql', 'query', 'milvus', 'mlserver']

def GetGlobalConfig(filename=_FILEPATH, sections=_CONFIG_SECTIONS):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    GlobalConfig = {}
    for section in sections:
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                GlobalConfig[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return GlobalConfig