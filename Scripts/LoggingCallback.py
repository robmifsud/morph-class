
class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """


    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn


    def on_epoch_end(self, epoch, logs={}):

        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)