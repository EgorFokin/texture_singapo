class EvalModule:
    def __init__(self,system_args):
        self.system_args = system_args
        

    def generate(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
    