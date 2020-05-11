class SetupFunctions:
    def __init__(self, settings):
        self.settings = settings

    def get_function_name(self, config):
        if type(config) is str:
            func_name = config
        else:
            func_name = config['func']
        return func_name

    def setup_function(self, object, key, config):
        feat = {
            "func": getattr(object, self.get_function_name(config)),
            "settings": {
                **self.settings,
                "key": key,
            }
        }
        if type(config) is not str:
            feat['settings'] = {**feat['settings'], **config.get('settings')}
        return feat

    def call_func(self, config, **attrs):
        return config['func'](**attrs, settings=config['settings'])
