from argparse import ArgumentParser
from loguru import logger

class Task:
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError
    
    def __getattr__(self, __name: str) -> any:
        # return none if attribute not exists
        if __name not in self.__dict__:
            return None
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{__name}'")

    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def launch(self):
        parser = ArgumentParser()
        parser = self.parse_task_args(parser)
        args, extras = parser.parse_known_args()
        # log the arguments
        logger.success(args)
        return self.run(**vars(args))